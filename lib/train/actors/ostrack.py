from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate

##################################################################
from lib.utils.nt_xent import NTXentLoss

class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        self.device = self._get_device()
        self.temperature = 0.5
        self.use_cosine_similarity = True
        self.nt_xent_criterion = NTXentLoss(self.device, self.bs,
                                            self.temperature, self.use_cosine_similarity)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\nRunning on:", device)

        if device == 'cuda':
            device_name = torch.cuda.get_device_name()
            print("The device name is:", device_name)
            cap = torch.cuda.get_device_capability(device=None)
            print("The capability of this device is:", cap, '\n')
        return device


    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1  # template_images: torch.Size([1, batch_size, 3, 128, 128])
        assert len(data['search_images']) == 1  # search_images: torch.Size([1, batch_size, 3, 256, 256])

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch_size, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch_size, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch_size, 3, 256, 256)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch_size, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        ###################################################################
        phrase_ids = data['phrase_ids'].permute(1, 0)            # 40*32-->32*40
        phrase_attnmask = data['phrase_attnmask'].permute(1, 0)  # 40*32-->32*40
        out_dict = self.net(template=template_list,        # 32*3*128*128 # (batch_size, 3, 128, 128)
                            search=search_img,             # 32*3*256*256 # (batch_size, 3, 256, 256)
                            phrase_ids =phrase_ids,        # 32*40
                            phrase_attnmask = phrase_attnmask, #32*40 # (batch_size, 40) 
                            ce_template_mask=box_mask_z,   # 32*64
                            ce_keep_rate=ce_keep_rate,     # 1
                            return_last_attn=False)
        
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        ###########################################################################################
        ## Multi-Modal Alignment
        # cross-modal alignment: loss_cma = 1/2(v2l + l2v)
        loss_cma, alpha_xl, alpha_zl = 0, 0.5, 0.5
        loss_cma_xl = torch.tensor(0.5).cuda() * self.nt_xent_criterion(pred_dict['vision_x_vectors'],
                                                                     pred_dict['language_vectors'])
        loss_cma_zl = torch.tensor(0.5).cuda() * self.nt_xent_criterion(pred_dict['vision_z_vectors'],
                                                                     pred_dict['language_vectors'])
        loss_cma = torch.tensor(alpha_xl).cuda() * loss_cma_xl\
                   + torch.tensor(alpha_zl).cuda() * loss_cma_zl  # alpha_xl*loss_cma_xl + alpha_zl*loss_cma_zl
        loss += loss_cma

        # intra-modal alignment: loss_ima 1/2(x2z + z2x)
        loss_ima, beta = 0, 1
        loss_ima = torch.tensor(0.5).cuda() * self.nt_xent_criterion(pred_dict['vision_x_vectors'],
                                                                     pred_dict['vision_z_vectors'])
        loss_ima = torch.tensor(beta).cuda() * loss_ima  # beta*loss_ima
        loss += loss_ima
        ###########################################################################################

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/cma": loss_cma.item(),
                      "Loss/ima": loss_ima.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
