import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

        ##############################################################
        ## resolution 256
        self.language_proj = nn.Linear(768, 768)
        self.language_xz_proj = nn.Linear(768, 256)
        self.vision_x_proj = nn.Linear(256*768, 256)
        self.vision_z_proj = nn.Linear(64*768, 256)

        ## resolution 384
        #self.language_proj = nn.Linear(768, 768)
        #self.language_xz_proj = nn.Linear(768, 256)
        #self.vision_x_proj = nn.Linear(576*768, 256)
        #self.vision_z_proj = nn.Linear(144*768, 256)

    ##################################################################
    def forward_features(self, z, x, language_embeddings,
                         mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        ##########################################################################################
        ## resolution 256: language_embeddings: 32*768*1*1-->32*768-->32*1*768-->language_proj:32*1*768-->normalize:32*1*768
        language_embeddings = F.normalize(self.language_proj(language_embeddings.squeeze().reshape(-1,1,768)), dim=2)
        ## language_embeddings = language_embeddings.squeeze().reshape(-1, 1, 768)  # no linear
        language_embeddings_x = language_embeddings.repeat(1, 256, 1) # 32*256*768
        language_embeddings_z = language_embeddings.repeat(1, 64, 1) # 32*64*768

        ## resolution 384: language_embeddings: 32*768*1*1-->32*768-->32*1*768
        #language_embeddings = F.normalize(self.language_proj(language_embeddings.squeeze().reshape(-1,1,768)), dim=2)
        ## language_embeddings = language_embeddings.squeeze().reshape(-1, 1, 768)  # no linear
        #language_embeddings_x = language_embeddings.repeat(1, 576, 1)
        #language_embeddings_z = language_embeddings.repeat(1, 144, 1)

        x = self.patch_embed(x)  # 32*3*256*256->32*256*768
        z = self.patch_embed(z)  # 32*3*128*128->32*64*768

        ###################################
        # Multi-Modal Alignment
        language_vectors = torch.squeeze(language_embeddings, 1)   # 32*1*768->32*768
        language_vectors = F.normalize(self.language_xz_proj(language_vectors), dim=1)  # 32*768->32*256

        # vision_x_vectors = torch.mean(x, dim=1) # 32*256*768 -> 32*768  # 32*256*768 -> flatten: 32*196608 -> vision_x_proj: 32*256 -> 32*256
        vision_x_vectors = F.normalize(self.vision_x_proj(torch.flatten(x, start_dim=1)), dim=1)  # 32*(256*768)->32*256

        # vision_z_vectors = torch.mean(z, dim=1)   # 32*64*768 -> 32*768  # 32*64*768 -> flatten: 32*49152 -> vision_z_proj: 32*256 -> 32*256
        vision_z_vectors = F.normalize(self.vision_z_proj(torch.flatten(z, start_dim=1)), dim=1)  # 32*(64*768)->32*256

        ###################################
        # Modal Mixup
        x = language_embeddings_x * x + x # 32*256*768
        z = language_embeddings_z * z + z # 32*64*768

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)


        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z # pos_embed_z:[1, 64, 768] -> 32*64*768
        x += self.pos_embed_x  # pos_embed_x:[1, 256, 768] ->32*256*768

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode) # -> 32*320*768
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x) # -> 32*320*768

        lens_z = self.pos_embed_z.shape[1] # 64
        lens_x = self.pos_embed_x.shape[1] # 256

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device) # [64]
        global_index_t = global_index_t.repeat(B, 1) # [64, 64]

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device) # [256]
        global_index_s = global_index_s.repeat(B, 1) # [64, 256]
        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x) # 32*320*768 -> 32*320*768
        lens_x_new = global_index_s.shape[1] # 256
        lens_z_new = global_index_t.shape[1] # 64

        z = x[:, :lens_z_new] # 32*64*768, 前64个为z
        x = x[:, lens_z_new:] # 32*256*768, 剩下的256个

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode) # 32*256*768

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1) # -> 32*320*768

        aux_dict = {
            "language_vectors": language_vectors,
            "vision_x_vectors": vision_x_vectors,
            "vision_z_vectors": vision_z_vectors,
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, language_embeddings, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        ##################################################################
        x, aux_dict = self.forward_features(z, x, language_embeddings, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
