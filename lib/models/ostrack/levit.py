# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License

import torch
import itertools
import lib.models.ostrack.levit_utils

from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model
from lib.models.ostrack.base_backbone import BaseBackbone
import torch.nn as nn
from lib.models.layers.patch_embed import PatchEmbed
import torch.nn.functional as F
from .utils import combine_tokens, recover_tokens

specification = {
    'LeViT_128S': {
        'C': '128_256_384', 'D': 16, 'N': '4_6_8', 'X': '2_3_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth'},
    'LeViT_128': {
        'C': '128_256_384', 'D': 16, 'N': '4_8_12', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth'},
    'LeViT_192': {
        'C': '192_288_384', 'D': 32, 'N': '3_5_6', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth'},
    'LeViT_256': {
        'C': '256_384_512', 'D': 32, 'N': '4_6_8', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth'},
    'LeViT_384': {
        'C': '384_512_768', 'D': 32, 'N': '6_9_12', 'X': '4_4_4', 'drop_path': 0.1,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth'},
}

__all__ = [specification.keys()]


@register_model
def LeViT_128S(num_classes=1000, distillation=True,
               pretrained=False, pretrained_cfg=None, fuse=False):
    return model_factory(**specification['LeViT_128S'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


@register_model
def LeViT_128(num_classes=1000, distillation=True,
              pretrained=False, pretrained_cfg=None, fuse=False):
    return model_factory(**specification['LeViT_128'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


@register_model
def LeViT_192(num_classes=1000, distillation=True,
              pretrained=False, pretrained_cfg=None, fuse=False):
    return model_factory(**specification['LeViT_192'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


@register_model
def LeViT_256(num_classes=1000, distillation=True,
              pretrained=False, pretrained_cfg=None, fuse=False):
    return model_factory(**specification['LeViT_256'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


@register_model
def LeViT_384(num_classes=1000, distillation=True,
              pretrained=False, pretrained_cfg=None, fuse=False):
    return model_factory(**specification['LeViT_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse)


FLOPS_COUNTER = 0


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = ((resolution + 2 * pad - dilation *
                          (ks - 1) - 1) // stride + 1)**2
        FLOPS_COUNTER += a * b * output_points * (ks**2) // groups

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution**2
        FLOPS_COUNTER += a * b * output_points

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * (resolution**4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**4)
        #attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution**4)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = (
        #     (q @ k.transpose(-2, -1)) * self.scale  # torch.Size([32, 4, 320, 320])
        #     +
        #     (self.attention_biases[:, self.attention_bias_idxs]  # attention_biases:torch.Size([4, 196]), attention_bias_idxs:torch.Size([196, 196]) --> torch.Size([4, 196, 196])
        #      if self.training else self.ab)
        # )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**2) * (resolution_**2)
        #attention * v
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * self.d

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
                               1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT(BaseBackbone):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 distillation=True,
                 drop_path=0,
                 drop_rate=0):
        super().__init__()
        global FLOPS_COUNTER

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0])
        num_patches = self.patch_embed.num_patches
        self.num_tokens = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        norm_layer=nn.LayerNorm
        self.norm = norm_layer(embed_dim)
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))
            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(torch.nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
        self.blocks = torch.nn.Sequential(*self.blocks)
        ##############################################################
        ## resolution 256
        # self.language_proj = nn.Linear(768, 768)
        self.language_proj = nn.Linear(768, 256)
        # self.language_xz_proj = nn.Linear(768, 256)
        self.language_xz_proj = nn.Linear(256, 256)
        # self.vision_x_proj = nn.Linear(256*768, 256)
        self.vision_x_proj = nn.Linear(256*256, 256)
        # self.vision_z_proj = nn.Linear(64*768, 256)
        self.vision_z_proj = nn.Linear(64*256, 256)

        ## resolution 384
        #self.language_proj = nn.Linear(768, 768)
        #self.language_xz_proj = nn.Linear(768, 256)
        #self.vision_x_proj = nn.Linear(576*768, 256)
        #self.vision_z_proj = nn.Linear(144*768, 256)

        ##################################################################
        # Classifier head
        # self.head = BN_Linear(
        #     embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        # if distillation:
        #     self.head_dist = BN_Linear(
        #         embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    # def forward(self, x):
    #     x = self.patch_embed(x)
    #     x = x.flatten(2).transpose(1, 2)
    #     x = self.blocks(x)
    #     x = x.mean(1)
    #     if self.distillation:
    #         x = self.head(x), self.head_dist(x)
    #         if not self.training:
    #             x = (x[0] + x[1]) / 2
    #     else:
    #         x = self.head(x)
    #     return x

    def forward_features(self, z, x, language_embeddings):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x) # 32*3*256*256->32*256*256
        z = self.patch_embed(z) # 32*3*128*128->32*64*256
        ##########################################################################################
        ## resolution 256: language_embeddings: 32*768*1*1--> 32*768-->32*1*768-->language_proj:32*1*256-->normalize:32*1*256
        language_embeddings = F.normalize(self.language_proj(language_embeddings.squeeze().reshape(-1,1,768)), dim=2)
        ## language_embeddings = language_embeddings.squeeze().reshape(-1, 1, 768)  # no linear
        language_embeddings_x = language_embeddings.repeat(1, 256, 1) # 32*256*256
        language_embeddings_z = language_embeddings.repeat(1, 64, 1) # 32*64*256

        ## resolution 384: language_embeddings: 32*768*1*1-->32*768-->32*1*768
        #language_embeddings = F.normalize(self.language_proj(language_embeddings.squeeze().reshape(-1,1,768)), dim=2)
        ## language_embeddings = language_embeddings.squeeze().reshape(-1, 1, 768)  # no linear
        #language_embeddings_x = language_embeddings.repeat(1, 576, 1)
        #language_embeddings_z = language_embeddings.repeat(1, 144, 1)

        # x = self.patch_embed(x)   # 32*3*256*256->32*256*768
        # z = self.patch_embed(z)   # 32*3*128*128->32*64*768

        ###################################
        # Multi-Modal Alignment
        language_vectors = torch.squeeze(language_embeddings, 1)   # 32*1*768->32*768
        language_vectors = F.normalize(self.language_xz_proj(language_vectors), dim=1)  # 32*768->32*256

        # 32*(256*256) -> 32*65536 -> vision_x_proj:32*256 ->normalize: 32*256
        vision_x_vectors = F.normalize(self.vision_x_proj(torch.flatten(x, start_dim=1)), dim=1)  # 32*(256*256)->32*256

        # 32*(64*256) -> 32*16384 -> vision_z_proj:32*256 ->normalize: 32*256
        vision_z_vectors = F.normalize(self.vision_z_proj(torch.flatten(z, start_dim=1)), dim=1)  # 32*(64*256)->32*256

        ###################################
        # Modal Mixup
        x = language_embeddings_x * x + x  # 32*256*256->32*256*256
        z = language_embeddings_z * z + z  # 32*64*256->32*64*256
        ###################################

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z  # 32*64*256
        x += self.pos_embed_x  # 32*256*256 

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)  # 32*64*256 + 32*256*256 -> 32*320*256 
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)  # 32*320*256

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        lens_z = self.pos_embed_z.shape[1] # 64
        lens_x = self.pos_embed_x.shape[1] # 256
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)  # 32*320*256

        # aux_dict = {"attn": None}
        aux_dict = {
            "language_vectors": language_vectors,
            "vision_x_vectors": vision_x_vectors,
            "vision_z_vectors": vision_z_vectors,
            "attn": None,
            # "removed_indexes_s": removed_indexes_s,  # used for visualization
        }
        # return self.norm(x), aux_dict
        return x, aux_dict
    
    def forward(self, z, x, language_embeddings, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        ##################################################################
        x, aux_dict = self.forward_features(z, x, language_embeddings)

        return x, aux_dict

def model_factory(C, D, X, N, drop_path, weights,
                  num_classes, distillation, pretrained, fuse):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = torch.nn.Hardswish
    model = LeViT(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        # down_ops=[
        #     #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
        #     ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
        #     ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        # ],
        down_ops=[],  # xyl
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    if fuse:
        utils.replace_batchnorm(model)

    return model

def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = model_factory(**kwargs, num_classes=1000, distillation=True, pretrained=False, fuse=False)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def levit_256(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(**specification['LeViT_256'], **kwargs)
    model = _create_vision_transformer('levit_256', pretrained=pretrained, **model_kwargs)
    return model
