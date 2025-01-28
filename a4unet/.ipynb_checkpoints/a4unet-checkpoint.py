# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:15:55 2023

@author: Ruoxin
"""
import math
import torch
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
from batchgenerators.augmentations.utils import pad_nd_image

from .unet_parts import *
from .utils import *
from .pyramid_block import DilatedSpatialPyramidPooling
from .grid_attention import GridAttentionBlock2D

from .nn import linear
from .nn import conv_nd
from .nn import checkpoint
from .nn import layer_norm
from .nn import avg_pool_nd
from .nn import zero_module
from .nn import normalization

from typing import Union, Tuple, List
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .fca import MultiSpectralAttentionLayer
from .sspp import SwinASPP
from .D_LKA.deformable_LKA import deformable_LKA_Attention


def create_a4unet_model(image_size, num_channels, num_res_blocks, num_classes, channel_mult="", learn_sigma=False, class_cond=False, use_checkpoint=False, 
                            attention_resolutions="16", in_ch=4, num_heads=1, num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False, 
                            dropout=0, resblock_updown=False):
    
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel_newpreview(image_size              = image_size,
                                in_channels             = in_ch,
                                model_channels          = num_channels,
                                out_channels            = 2, 
                                num_res_blocks          = num_res_blocks,
                                attention_resolutions   = tuple(attention_ds),
                                dropout                 = dropout,
                                channel_mult            = channel_mult,
                                num_classes             = num_classes,
                                use_checkpoint          = use_checkpoint,
                                num_heads               = num_heads,
                                num_head_channels       = num_head_channels,
                                num_heads_upsample      = num_heads_upsample,
                                use_scale_shift_norm    = use_scale_shift_norm,
                                resblock_updown         = resblock_updown,)


class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class deformableLKABlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) # build_norm_layer(norm_cfg, dim)[1]
        self.attn = deformable_LKA_Attention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim) # build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, C, H, W = x.shape
        N = H * W

        y = x.permute(0, 2, 3, 1) # b h w c, because norm requires this
        y = self.norm1(y)

        y = y.permute(0, 3, 1, 2) # b c h w, because attn requieres this
        y = self.attn(y)
        y = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
        #                       * self.attn(self.norm1(x)))

        y = x.permute(0, 2, 3, 1) # b h w c, because norm requires this
        y = self.norm2(y)

        y = y.permute(0, 3, 1, 2) # b c h w, because attn requieres this
        y = self.mlp(y)
        y = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * y
        y = self.drop_path(y)
        x = x + y
        # x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
        #                       * self.mlp(self.norm2(x)))

        # x = x.view(B, C, N).permute(0, 2, 1)
        
        # print("LKA return shape: {}".format(x.shape))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class UNetModel_newpreview(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), 
                 conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1, 
                 use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.n_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.n_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim),
                                        nn.SiLU(),
                                        linear(time_embed_dim, time_embed_dim))

        # 初始输入层 TimestepEmbedSequential的作用是给予基础层时序属性,
        # 通过这个时序属性将其与普通层区分开
        # 维度、输入通道数、输出通道数、核心尺寸、步长(默认为1)、填充值
        self.input_blocks = nn.ModuleList([nn.Sequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])

        c2wh = dict([(128, 56), (256, 28), (384, 14), (512, 7)])
        self.FCA4 = MultiSpectralAttentionLayer(model_channels * 4, c2wh[model_channels], c2wh[model_channels],
                                                reduction=16, freq_sel_method='top16')
        self.FCA3 = MultiSpectralAttentionLayer(model_channels * 3, c2wh[model_channels], c2wh[model_channels],
                                                reduction=16, freq_sel_method='top16')
        self.FCA2 = MultiSpectralAttentionLayer(model_channels * 2, c2wh[model_channels], c2wh[model_channels],
                                                reduction=16, freq_sel_method='top16')
        self.FCA1 = MultiSpectralAttentionLayer(model_channels, c2wh[model_channels], c2wh[model_channels],
                                                reduction=16, freq_sel_method='top16')
        self.SA = SpatialAttention()

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch, ds = model_channels, 1 # downsample序号
        # print("model ch: ", model_channels)
        # 编码器组块 level是channel_mult内元素的序号,
        # mult是其内元素, 编码器一共len(channel_mult)个卷积块
        # level = 0, 1, 2, 3
        self.DLKA_blocks = nn.ModuleList([])

        for level, mult in enumerate(channel_mult):
            for _ in range(2):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels,
                                   dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                # 注意上一行这个ResBlock根本没有输入upsample和downsample参数,
                # 说明两参数默认为False, ResBlock内采用恒等映射
                ch = mult * model_channels
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
            
            # 记录每层Downsample前的特征层的通道数
            input_block_chans.append(ch)
            
            # 注意这个if结构并没有在双层for循环内,
            # 而是在单层for循环内, 说明是在每个level最后添加的组件
            # 最后一个level不额外添加下采样层
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.DLKA_blocks.append(deformableLKABlock(dim=out_ch).to('cuda:0'))
                self.input_blocks.append(nn.Sequential(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, 
                                                                use_scale_shift_norm=use_scale_shift_norm, down=True)
                        if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                # 特别注意这个if else结构, 乍一看像是添加两个组件, 测试发现实际上是添加一个组件,
                # 当resblock_updown为True时添加前面的ResBlock, 当为False时添加后面的Downsample
                ch = out_ch
                ds *= 2
                self._feature_size += ch
            if level == len(channel_mult) - 1:
                self.DLKA_blocks.append(deformableLKABlock(dim=ch).to('cuda:0'))
        
        # 中间层组块
        self.middle_block = nn.Sequential(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
                                          AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order),
                                          ResBlock(ch, time_embed_dim, dropout, out_channels=1024, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        #################### ASPP ##################
        # # Feature fusion.
        # # Add a feature pyramid layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.aspp = SwinASPP(
            input_size=8,
            input_dim=1024,
            out_dim=512,
            cross_attn='CBAM',
            depth=2,    # BasicLayer 中 Transformer 层的数量。
            num_heads=32, # Transformer 中注意力头的数量。
            mlp_ratio=4, # Transformer 中 MLP（多层感知机）部分输出维度相对于输入维度的倍数。
            qkv_bias=True, # 在注意力计算中，是否允许 Query、Key、Value 的偏置项。
            qk_scale=None,  # 在注意力计算中，对 Query、Key 的缩放因子。
            drop_rate=0.,  # 通用的丢弃率，可以应用到多个部分，例如 MLP、Dropout 等。
            attn_drop_rate=0.,  # 注意力计算中的丢弃率。
            drop_path_rate=0.1,  # DropPath（一种用于随机删除网络中的路径） 的概率。
            norm_layer=nn.LayerNorm,  # 规范化层的类型，可以是 PyTorch 中的规范化层类。
            aspp_norm=False,   # 是否在 ASPP 模块中使用规范化。
            aspp_activation='relu', # ASPP 模块中的激活函数类型。
            start_window_size=7, # ASPP 模块中可能的窗口大小的起始值。
            aspp_dropout=0.1,  # ASPP 模块中的丢弃率。
            downsample=None, # 是否使用下采样，感觉好像没什么用。
            use_checkpoint=True # 是否使用模型参数的检查点，用于减少 GPU 内存使用。
        )
        ############################################
        ################## Gate ##################
        # Spatial attention -- Attention gate.
        self.gating = UnetGridGatingSignal2(512, 512, kernel_size=1)
        # attention blocks
        self.attention_dsample = (2, 2)
        self.nonlocal_mode = 'concatenation'
        ##########################################
        
        self._feature_size += ch

        # 解码器组块 level与mult均为倒序, 解码器一共len(channel_mult)个卷积块, 与编码器一样
        self.output_blocks = nn.ModuleList([])
        self.layer_gate_attn = []

        # 与编码器for循环的区别就是倒序而已
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # 注意解码器每层3个卷积块, 而编码器每层2个卷积块
            for i in range(3):
                # ich = input_block_chans.pop()
                if i == 0:
                    ich = input_block_chans.pop()
                    layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                # elif i != 3:
                #     layers = [nn.Conv2d(ch * 2, ch, 1, bias=False),
                #               ResBlock(ch, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                else:
                    layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]

            ch = model_channels * mult

            # 末尾层不额外加卷积层, 其余每层都在末尾额外增加1个Upsample
            if level and i == num_res_blocks:
                out_ch = ch # 512, 384, 256, 128.
                
                layers.append(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                       use_checkpoint=use_checkpoint,
                                       use_scale_shift_norm=use_scale_shift_norm, up=True)
                                       if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))

                ########## Attention Gate #############
                self.layer_gate_attn.append(GridAttentionBlock2D(in_channels=ch, emb_channels=time_embed_dim, gating_channels=512,
                                            inter_channels=ch, sub_sample_factor=self.attention_dsample,
                                            mode=self.nonlocal_mode).to('cuda:0'))
                #######################################

                ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        # 最终输出层
        self.out = nn.Sequential(normalization(ch),
                                 nn.SiLU(),
                                 conv_nd(dims, model_channels , out_channels, 3, padding=1))
        

    def forward(self, x, y=None):
        # 1. 输入处理
        h = x.type(self.dtype)

        # 2. 编码器前向传播
        encoder_features = []
        ind_dlka = 0

        for ind, module in enumerate(self.input_blocks):
            h = module(h)
            if (ind - 2) % 3 == 0:
                h = self._apply_dlka(h, ind_dlka)
                encoder_features.append(h)
                ind_dlka += 1

        # 3. 中间层处理
        h = self.middle_block(h)

        # ASPP 处理
        h = h.permute(0, 2, 3, 1)
        h = self.aspp(h)
        h = h.permute(0, 3, 1, 2)

        # 空间注意力门控
        gating = self.gating(h)

        # 4. 解码器前向传播
        for ind, module in enumerate(self.output_blocks):
            h = module(h)
            if ind != 3:
                h = self.layer_gate_attn[ind](h, gating)
                # 应用频率注意力
                if ind == 0:
                    h = self.FCA4(h)
                elif ind == 1:
                    h = self.FCA3(h)
                elif ind == 2:
                    h = self.FCA2(h)
            else:
                h = self.FCA1(h)
            h = self.SA(h) * h

        # 5. 输出处理
        return self.out(h.type(x.dtype))

    def _apply_dlka(self, h, ind):
        """应用 DLKA block 处理"""
        _, _, H, W = h.shape
        return self.DLKA_blocks[ind](h, H, W)


class ResBlock(nn.Module): # 这个残差块也继承了时序块的属性, 就是单纯将其与正常模块区分开
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False,
                 use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # 输入层
        self.in_layers = nn.Sequential(normalization(channels),
                                       nn.SiLU(),
                                       conv_nd(dims, channels, self.out_channels, 3, padding=1))

        # 用于确定该卷积块是否属于level中最后一个卷积块的Flag
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        # 输出层
        self.out_layers = nn.Sequential(normalization(self.out_channels),
                                        nn.SiLU(),
                                        nn.Dropout(p=dropout),
                                        conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x): # (self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # (x,)这个逗号极端重要, 如果没有这个逗号那么就无法构成tuple, 后续会出错
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        # 当该卷积块不处于level中最后一个卷积块那么此参数为False
        if self.updown:
            # [:-1]是输入除最后一个元素外其他元素, [-1]是输出最后一个元素
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        # use_scale_shift_norm=False
        if self.use_scale_shift_norm:
            # out_layers[0]是GroupNorm32, out_layers[1:]是 SiLU+Conv2d
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            h = out_rest(h)
        else:
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(oup),
                         nn.ReLU(inplace=True))

def conv_dw(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), # dw
                         nn.BatchNorm2d(inp),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(inp, oup, 1, 1, 0, bias=False), # pw
                         nn.BatchNorm2d(oup),
                         nn.ReLU(inplace=True))


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1: # 当为-1时只使用1个自注意力头处理所有通道
            self.num_heads = num_heads
        else:
            assert (channels % num_head_channels == 0), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1) # 1维卷积, 输入通道数, 输出通道数, 核心为1, 步长默认为1, 填充默认为0; 单个卷积核通道数为channels, 一共有3个卷积核
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape # *spatial = height, width
        x = x.reshape(b, c, -1) # x.shape = [batch, channels, height×width]
        qkv = self.qkv(self.norm(x)) # qkv.shape = batch, 3×channels, height×width; 一维卷积输出序列计算公式为output_L=input_L-kernel+padding
        h = self.attention(qkv) # 标准多头自注意力, 其中自注意力头数量为4, h.shape=[bs, channel, height×width]
        h = self.proj_out(h) # 线性投影层, h.shape=[bs, channel, height×width]
        return (x + h).reshape(b, c, *spatial) # 将原特征层与经过MSA处理的特征层相加


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(model,
                                    inputs=(inputs, timestamps),
                                    custom_ops={QKVAttention: QKVAttention.count_flops})
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape # [bs, 3×channel, height×width]
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads) # ch = width / 3 / n_heads
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1) # q.shape, k.shape, v.shape = [bs×n_heads, channel/n_heads, height×width]
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale) # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv): # 输入张量为[batch, patch_num, length], 其中length=channel×height×width
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1) # 将经过线性投影的张量分成Query, Key, Value
        scale = 1 / math.sqrt(math.sqrt(ch)) # 计算根号dk用于归一化数值
        weight = th.einsum("bct,bcs->bts", 进行QK相乘
                           (q * scale).view(bs * self.n_heads, ch, length),
                           (k * scale).view(bs * self.n_heads, ch, length))  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype) # 单个Q对所有K进行向量相乘得到的值进行Softmax得到权重
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)) # 对V和Softmax得到的权重进行加权相加得到新V
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
    

def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)