import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

from .AbaViT_patch_embed import PatchEmbed
from timm.models.layers import Mlp, DropPath
from .AbaViT_utils import get_distribution_target, combine_tokens, recover_tokens
from torch.autograd import Variable
import torch.nn.functional as F
from einops import rearrange

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

_logger = logging.getLogger(__name__)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

# --------------------------------------------------------------------------------------------------------------------------------- xyl 稀疏注意力
##  Top-K Sparse Attention (TKSA)
class TKSA_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TKSA_Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        if len(x.size()) == 3:  # bs, 序列长度，embed_dim
            x = rearrange(x, 'b c (h w) -> b c h w', head=self.num_heads)
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
# --------------------------------------------------------------------------------------------------------------------------------- xyl 稀疏注意力

class Masked_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mask=None, masked_softmax_bias=-1000.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask = mask
        self.masked_softmax_bias = masked_softmax_bias

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask.view(mask.shape[0], 1, 1, mask.shape[1]) * self.masked_softmax_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block_ACT(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None, index=-1, num_patches=197):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Masked_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.act_mode = args.act_mode
        assert self.act_mode in {1, 2, 3, 4}

        self.index=index
        self.args = args

        if self.act_mode == 4:
            self.sig = torch.sigmoid
        else:
            print('Not supported yet.')
            exit()

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


    def forward_act(self, x, mask=None):

        bs, token, dim = x.shape

        if mask is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)))

        if self.act_mode==4:
            gate_scale, gate_center = self.args.gate_scale, self.args.gate_center
            halting_score_token = self.sig(x[:,:,0] * gate_scale - gate_center)
            halting_score = [-1, halting_score_token]
        else:
            print('Not supported yet.')
            exit()

        return x, halting_score

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', args=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_embed_z = nn.Parameter(torch.zeros(1, 64, 192))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, 256, 192))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(*[
            Block_ACT(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, args=args, index=i, num_patches=self.patch_embed.num_patches+1)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.eps = 0.01
        for block in self.blocks:
            if args.act_mode == 1:
                torch.nn.init.constant_(block.act_mlp.fc2.bias.data, -1. * args.gate_center)

        self.args = args

        self.rho = None
        self.counter = None
        self.batch_cnt = 0

        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.rho_token_weight = None
        self.counter_token = None
        self.total_token_cnt = int((128/patch_size)**2 + (256/patch_size)**2) + self.num_tokens

        if args.distr_prior_alpha >0. :
            self.distr_target = torch.Tensor(get_distribution_target(standardized=True, target_depth=5)).cuda()
            self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        self.cat_mode = 'direct'

        ##############################################################
        ## resolution 256
        # self.language_proj = nn.Linear(768, 768)
        self.language_proj = nn.Linear(768, 192)
        # self.language_xz_proj = nn.Linear(768, 256)
        self.language_xz_proj = nn.Linear(192, 256)
        # self.vision_x_proj = nn.Linear(256*768, 256)
        self.vision_x_proj = nn.Linear(256*192, 256)
        # self.vision_z_proj = nn.Linear(64*768, 256)
        self.vision_z_proj = nn.Linear(64*192, 256)

        ## resolution 384
        #self.language_proj = nn.Linear(768, 768)
        #self.language_xz_proj = nn.Linear(768, 256)
        #self.vision_x_proj = nn.Linear(576*768, 256)
        #self.vision_z_proj = nn.Linear(144*768, 256)
        ##################################################################
    # --------------------------------------------------------------------------------------------------------------------------------- xyl 
        self.tksa = TKSA_Attention(dim=192, num_heads=8, bias=False)
    # =================================================================================== xyl 
        self.img_dim = 192
        self.text_dim = 192
        self.fusion_visual_textual = nn.MultiheadAttention(
                embed_dim=self.img_dim,
                num_heads=4,
                dropout=0,
            )
        self.fusion_fc = nn.Linear(self.text_dim, self.img_dim)
        self.fusion_drop = nn.Dropout(p=0.1)

    def cross_modal_fusion(self, vis_feat, text_feat, mode):
    # def cross_modal_fusion(self, vis_feat, text_feat, b, t, mode):
        # b, t = global_img.size()[:2]  # b t c h w
        if mode == 'cascade attention':
            assert len(text_feat.size()) == 3
            # get textual embeddings
            # text_feat = text_feat.unsqueeze(1)  # [b,l,c]->[b,1,l,c]
            # text_feat = text_feat.repeat([1, t, 1, 1])
            # text_feat = rearrange(text_feat, 'b t l c -> (b t) l c')
            text_feat = self.fusion_fc(text_feat)  # 32*256*192 -> 32*256*192
            # text_feat = rearrange(text_feat, 'bt l c -> l bt c')  # 32*256*192 -> 256*32*192
            # fusion
            fused_feat = self.fusion_visual_textual(
                query=vis_feat,  # 三维token的顺序是(batch_size, sequence_length, embed_dim)
                key=text_feat,
                value=text_feat,
            )[0]
            # vis_feat = vis_feat * fused_feat # 32*256*192 -> 32*256*192
            vis_feat = vis_feat * fused_feat + vis_feat # 32*256*192 -> 32*256*192, 残差
            # vis_feat = rearrange(vis_feat, 'l bt c -> bt c l')
            return vis_feat
    # =================================================================================== xyl 

    # def forward_features_act_token(self, z, x, t_mask=None, s_mask=None):
    def forward_features_act_token(self, z, x, language_embeddings, t_mask=None, s_mask=None):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        if not t_mask is None:
            t_mask = t_mask[:,0:t_mask.shape[1]:self.patch_size,0:t_mask.shape[2]:self.patch_size]
            s_mask = s_mask[:,0:s_mask.shape[1]:self.patch_size,0:s_mask.shape[2]:self.patch_size]
            t_mask1 = 1-t_mask
            s_mask1 = 1-s_mask
            t_mask = 1.5*t_mask1 + 1*t_mask
            s_mask = 1.5*s_mask1 + 1*s_mask
            t_mask = t_mask.view(t_mask.shape[0],-1)
            s_mask = s_mask.view(s_mask.shape[0],-1)

        x = self.patch_embed(x) # 32*3*256*256->32*256*192
        z = self.patch_embed(z) # 32*3*128*128->32*64*192

        x = x + self.pos_embed_x # 32*256*192
        z = z + self.pos_embed_z # 32*64*192

        ##########################################################################################
        ## resolution 256: language_embeddings: 32*768*1*1--> 32*768-->32*1*768-->language_proj:32*1*192-->normalize:32*1*192
        language_embeddings = F.normalize(self.language_proj(language_embeddings.squeeze().reshape(-1,1,768)), dim=2)
        ## language_embeddings = language_embeddings.squeeze().reshape(-1, 1, 768)  # no linear
        language_embeddings_x = language_embeddings.repeat(1, 256, 1) # 32*256*192
        language_embeddings_z = language_embeddings.repeat(1, 64, 1) # 32*64*192

        ## resolution 384: language_embeddings: 32*768*1*1-->32*768-->32*1*768
        #language_embeddings = F.normalize(self.language_proj(language_embeddings.squeeze().reshape(-1,1,768)), dim=2)
        ## language_embeddings = language_embeddings.squeeze().reshape(-1, 1, 768)  # no linear
        #language_embeddings_x = language_embeddings.repeat(1, 576, 1)
        #language_embeddings_z = language_embeddings.repeat(1, 144, 1)

        # x = self.patch_embed(x)   # 32*3*256*256->32*256*768
        # z = self.patch_embed(z)   # 32*3*128*128->32*64*768

        ###################################
        # Multi-Modal Alignment
        language_vectors = torch.squeeze(language_embeddings, 1)   # 32*1*192->32*192
        language_vectors = F.normalize(self.language_xz_proj(language_vectors), dim=1)  # 32*192->32*256

        # 32*(256*192) -> 32*49152 -> vision_x_proj:32*256 ->normalize: 32*256
        vision_x_vectors = F.normalize(self.vision_x_proj(torch.flatten(x, start_dim=1)), dim=1)  # 32*(256*256)->32*256

        # 32*(64*192) -> 32*12288 -> vision_z_proj:32*256 ->normalize: 32*256
        vision_z_vectors = F.normalize(self.vision_z_proj(torch.flatten(z, start_dim=1)), dim=1)  # 32*(64*256)->32*256

        ###################################
        # Modal Mixup
        CA_Mixup = True
        TKSA = True
        if not CA_Mixup:
            x = language_embeddings_x * x + x  # 32*256*192->32*256*192
            z = language_embeddings_z * z + z  # 32*64*192->32*64*192
        elif CA_Mixup:
            x = self.cross_modal_fusion(x, language_embeddings_x, 'cascade attention')
            z = self.cross_modal_fusion(z, language_embeddings_z, 'cascade attention')
        elif TKSA:
            x = self.tksa(x)
        ###################################
        ##########################################################################################

        x = combine_tokens(z, x, mode=self.cat_mode)

        if not t_mask is None:
            self.rho_token_weight = combine_tokens(t_mask, s_mask, mode=self.cat_mode)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # cls_token: torch.Size([1, 1, 192]) # 第一维是bs，训练是32，测试是1
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # x: torch.Size([1, 321, 192])
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x)
        bs = x.size()[0]

        if self.c_token is None or bs != self.c_token.size()[0]:
            self.c_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.R_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.mask_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.rho_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.counter_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

        c_token = self.c_token.clone()
        R_token = self.R_token.clone()
        mask_token = self.mask_token.clone()
        self.rho_token = self.rho_token.detach() * 0.
        self.counter_token = self.counter_token.detach() * 0 + 1.
        output = None
        out = x

        if self.args.distr_prior_alpha>0.:
            self.halting_score_layer = []

        for i, l in enumerate(self.blocks):
            out.data = out.data * mask_token.float().view(bs, self.total_token_cnt, 1)
            block_output, h_lst = l.forward_act(out, 1.-mask_token.float())  # out: torch.Size([1, 321, 192]), block_output: torch.Size([1, 321, 192])

            if self.args.distr_prior_alpha>0.:
                self.halting_score_layer.append(torch.mean(h_lst[1][1:]))

            out = block_output.clone()

            _, h_token = h_lst

            block_output = block_output * mask_token.float().view(bs, self.total_token_cnt, 1)

            if i==len(self.blocks)-1:
                h_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

            c_token = c_token + h_token
            self.rho_token = self.rho_token + mask_token.float()

            reached_token = c_token > 1 - self.eps
            reached_token = reached_token.float() * mask_token.float()
            delta1 = block_output * R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs, self.total_token_cnt, 1)
            self.rho_token = self.rho_token + R_token * reached_token

            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float()
            R_token = R_token - (not_reached_token.float() * h_token)
            delta2 = block_output * h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs, self.total_token_cnt, 1)

            self.counter_token = self.counter_token + not_reached_token

            mask_token = c_token < 1 - self.eps

            if output is None:
                output = delta1 + delta2  # output: torch.Size([1, 321, 192])
            else:
                output = output + (delta1 + delta2)

        x = self.norm(output)


        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        # aux_dict = {"attn": None}
        aux_dict = {
            "language_vectors": language_vectors,
            "vision_x_vectors": vision_x_vectors,
            "vision_z_vectors": vision_z_vectors,
            "attn": None,
            # "removed_indexes_s": removed_indexes_s,  # used for visualization
        }
        return x, aux_dict

    def forward(self, z, x, language_embeddings, t_mask=None, s_mask=None, **kwargs):
        if self.args.act_mode == 4:
            x, aux_dict = self.forward_features_act_token(z,x, language_embeddings, t_mask=t_mask, s_mask=s_mask)
        else:
            print('Not implemented yet, please specify for token act.')
            exit()

        return x, aux_dict


# def torch_load_legacy(path):
#     """Load network with legacy environment."""

#     # Setup legacy env (for older networks)
#     _setup_legacy_env()

#     # Load network
#     checkpoint_dict = torch.load(path, map_location='cpu')

#     # Cleanup legacy
#     _cleanup_legacy_env()

#     return checkpoint_dict

from argparse import Namespace
__all__ = [
    'abavit_patch16_224'
]
def abavit_patch16_224(pretrained=False):
    kwargs = {'num_classes': 1000, 'drop_rate': 0.0, 'drop_path_rate': 0.1}
    model_kwargs = {'act_mode': 4, 'gate_scale': 10.0, 'gate_center': 30.0,'distr_prior_alpha':0.01}
    model_kwargs = Namespace(**model_kwargs)

    kwargs['args'] = model_kwargs
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            # missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            
            # copy from MKDNet            
            backbone_dict = model.state_dict()
            pretrain_dict = {k[len('backbone.'):]: v for k, v in checkpoint.items() if
                                k[len('backbone.'):] in backbone_dict}
            missing_keys, unexpected_keys = model.load_state_dict(pretrain_dict, strict=False)

            print("===========missing_keys===========: \n", missing_keys)
            print("=========unexpected_keys==========: \n", unexpected_keys)
            # missing_keys, unexpected_keys = model.load_state_dict(checkpoint["backbone"], strict=False)
            print('Load pretrained model from: ' + pretrained)
    
    return model








