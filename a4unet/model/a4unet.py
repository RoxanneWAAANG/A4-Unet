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
                            dropout=0, resblock_updown=False, use_fp16=False, use_new_attention_order=False):
    
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
                                use_fp16                = use_fp16,
                                num_heads               = num_heads,
                                num_head_channels       = num_head_channels,
                                num_heads_upsample      = num_heads_upsample,
                                use_scale_shift_norm    = use_scale_shift_norm,
                                resblock_updown         = resblock_updown,
                                use_new_attention_order = use_new_attention_order)


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
                 use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, high_way = True):
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
        
        # 删除Condition UNet架构
        high_way = False
        
        if high_way:
            features = 32
            self.hwm = Generic_UNet(self.in_channels - 1, features, 1, 5, anchor_out=True, upscale_logits=True)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def load_part_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                    continue
            if isinstance(param, th.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
    
    def enhance(self, c, h):
        cu = layer_norm(c.size()[1:])(c)
        hu = layer_norm(h.size()[1:])(h)
        return cu * hu * h
    
    def highway_forward(self,x, hs = None):
        return self.hwm(x, hs = None)

    def forward(self, x, y=None): # (self, x, timesteps, y=None):
        hs=[]
        
        # 做一些前期准备工作: 转换输入值至固定type, 从输入值中取得图片, 取得anchor与calibration
        h = x.type(self.dtype)
        
        # 编码器模块 从此输入值正式进入去噪网络
        ind_DLKA = 0
        for ind, module in enumerate(self.input_blocks):
            # 向第1个解码器卷积块输入anch, 注意是在输入层后, 第1个解码器卷积块前拼接anch
            h = module(h)
            if (ind - 2) % 3 == 0:
                _, _, H, W = h.shape
                h = self.DLKA_blocks[ind_DLKA](h, H, W)
                # 将特征层逐卷积块记录, 用于与解码器特征层concat, 注意一共4层, 12个卷积块
                hs.append(h)
                ind_DLKA += 1
        
        # 中间嵌入层模块
        h = self.middle_block(h)
        ################ ASPP ##################
        # # Feature fusion -- ASPP.
        h = h.permute(0, 2, 3, 1)
        h = self.aspp(h)
        h = h.permute(0, 3, 1, 2)

        # Spatial attention -- Attention gate.
        gating = self.gating(h)
        # h = self.bottleneck(h)
        ########################################

        # 解码器模块
        for ind, module in enumerate(self.output_blocks):
            
            h = module(h)
            if ind != 3:
                # h = th.cat([h, hs.pop()], dim=1)
                # h = module(h)
                h = self.layer_gate_attn[ind](h, gating)
                if ind == 0:
                    h = self.FCA4(h)
                if ind == 1:
                    h = self.FCA3(h)
                if ind == 2:
                    h = self.FCA2(h)
            else:
                h = self.FCA1(h)
            h = self.SA(h) * h
            
        h = h.type(x.dtype) # 类型转换

        # 最终输出层
        out = self.out(h) # 输出层是一个卷积层, 标准卷积层包括Conv、BatchNorm、SiLU三个操作
        return out # return out, cal


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


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads_channels: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1) # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1) # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype) # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


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

class MobBlock(nn.Module):
    def __init__(self,ind):
        super().__init__()

        if ind == 0:
            self.stage = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 1),
            conv_dw(128, 128, 1))
        elif ind == 1:
            self.stage  = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1))
        elif ind == 2:
            self.stage = nn.Sequential(
            conv_dw(256, 256, 2),
            conv_dw(256, 256, 1))
        else:
            self.stage = nn.Sequential(
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1))

    def forward(self,x):
        return self.stage(x)

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

class FFParser(nn.Module):
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device.type == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # if we have 5 pooling then our patch size must be divisible by 2**5
        self.input_shape_must_be_divisible_by = None  # for example in a 2d network that does 5 pool in x and 6 pool
        # in y this would be (32, 64)

        # we need to know this because we need to know if we are a 2d or a 3d netowrk
        self.conv_op = None  # nn.Conv2d or nn.Conv3d

        # this tells us how many channels we have in the output. Important for preallocation in inference
        self.num_classes = None  # number of channels in the output

        # depending on the loss, we do not hard code a nonlinearity into the architecture. To aggregate predictions
        # during inference, we need to apply the nonlinearity, however. So it is important to let the newtork know what
        # to apply in inference. For the most part this will be softmax
        self.inference_apply_nonlin = lambda x: x  # softmax_helper

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

    def predict_3D(self, 
                   x: np.ndarray, 
                   do_mirroring: bool, 
                   mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, 
                   patch_size: Tuple[int, ...] = None, 
                   regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, 
                   pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, 
                   all_in_gpu: bool = False,
                   verbose: bool = True, 
                   mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv3d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                     verbose=verbose)
                    else:
                        res = self._internal_predict_3D_3Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose)
                elif self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_2Dconv_tiled(x, patch_size, do_mirroring, mirror_axes, step_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, False)
                    else:
                        res = self._internal_predict_3D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res

    def predict_2D(self, 
                   x, 
                   do_mirroring: bool, 
                   mirror_axes: tuple = (0, 1, 2), 
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, 
                   patch_size: tuple = None, 
                   regions_class_order: tuple = None,
                   use_gaussian: bool = False, 
                   pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, 
                   all_in_gpu: bool = False,
                   verbose: bool = True, 
                   mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaler than 1. Otherwise there will be a gap between consecutive predictions'

        if self.conv_op == nn.Conv3d:
            raise RuntimeError("Cannot predict 2d if the network is 3d. Dummy.")

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3) for a 2d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if max(mirror_axes) > 1:
                raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 3, "data must have shape (c,x,y)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_2D_2Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, 
                                                                     use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose)
                                                                     
                    else:
                        res = self._internal_predict_2D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, verbose)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res

    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999 # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

    def _internal_predict_3D_3Dconv_tiled(self,
                                          x: np.ndarray,
                                          step_size: float,
                                          do_mirroring: bool,
                                          mirror_axes: tuple,
                                          patch_size: tuple,
                                          regions_class_order: tuple,
                                          use_gaussian: bool,
                                          pad_border_mode: str,
                                          pad_kwargs: dict,
                                          all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose: print("done")
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

            #predict on cpu if cuda not available
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())

        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring,
                        gaussian_importance_map)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        aggregated_results /= aggregated_nb_of_predictions
        del aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = aggregated_results.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = aggregated_results.detach().cpu().numpy()
            else:
                class_probabilities_here = aggregated_results
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            aggregated_results = aggregated_results.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, aggregated_results

    def _internal_predict_2D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 3, "x must be (c, x, y)"

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_2D_2Dconv'
        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True, self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_2D(data[None], mirror_axes, do_mirroring, None)[0]

        slicer = tuple([slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) - (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_predict_3D_3Dconv(self, 
                                    x: np.ndarray, 
                                    min_size: Tuple[int, ...], 
                                    do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), 
                                    regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", 
                                    pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to run _internal_predict_3D_3Dconv'
        
        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_3D(data[None], mirror_axes, do_mirroring, None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) - (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        x = maybe_to_torch(x)
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, ))))
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_maybe_mirror_and_pred_2D(self, 
                                           x: Union[np.ndarray, torch.tensor], 
                                           mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(x.shape) == 4, 'x must be (b, c, x, y)'

        x = maybe_to_torch(x)
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_predict_2D_2Dconv_tiled(self, 
                                          x: np.ndarray, 
                                          step_size: float, 
                                          do_mirroring: bool, 
                                          mirror_axes: tuple,
                                          patch_size: tuple, 
                                          regions_class_order: tuple, 
                                          use_gaussian: bool,
                                          pad_border_mode: str, 
                                          pad_kwargs: dict, 
                                          all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 3, "x must be (c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape # still c, x, y

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_2d is None or not all([i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_2d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_2d = gaussian_importance_map
                self._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_2d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D(
                    data[None, :, lb_x:ub_x, lb_y:ub_y], mirror_axes, do_mirroring,
                    gaussian_importance_map)[0]

                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple([slice(0, aggregated_results.shape[i]) for i in range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_predict_3D_2Dconv(self, 
                                    x: np.ndarray, 
                                    min_size: Tuple[int, int], 
                                    do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1), 
                                    regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", 
                                    pad_kwargs: dict = None,
                                    all_in_gpu: bool = False, 
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv(x[:, s], min_size, do_mirroring, mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    def predict_3D_pseudo3D_2Dconv(self, 
                                   x: np.ndarray, 
                                   min_size: Tuple[int, int], 
                                   do_mirroring: bool,
                                   mirror_axes: tuple = (0, 1), 
                                   regions_class_order: tuple = None,
                                   pseudo3D_slices: int = 5, 
                                   all_in_gpu: bool = False,
                                   pad_border_mode: str = "constant", 
                                   pad_kwargs: dict = None,
                                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        assert pseudo3D_slices % 2 == 1, "pseudo3D_slices must be odd"
        extra_slices = (pseudo3D_slices - 1) // 2

        shp_for_pad = np.array(x.shape)
        shp_for_pad[1] = extra_slices

        pad = np.zeros(shp_for_pad, dtype=np.float32)
        data = np.concatenate((pad, x, pad), 1)

        predicted_segmentation = []
        softmax_pred = []
        for s in range(extra_slices, data.shape[1] - extra_slices):
            d = data[:, (s - extra_slices):(s + extra_slices + 1)]
            d = d.reshape((-1, d.shape[-2], d.shape[-1]))
            pred_seg, softmax_pres=self._internal_predict_2D_2Dconv(d, min_size, do_mirroring, mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred

    def _internal_predict_3D_2Dconv_tiled(self, 
                                          x: np.ndarray, 
                                          patch_size: Tuple[int, int], 
                                          do_mirroring: bool,
                                          mirror_axes: tuple = (0, 1), 
                                          step_size: float = 0.5,
                                          regions_class_order: tuple = None, 
                                          use_gaussian: bool = False,
                                          pad_border_mode: str = "edge", 
                                          pad_kwargs: dict =None,
                                          all_in_gpu: bool = False,
                                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        assert len(x.shape) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled(
                x[:, s], step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels, conv_op=nn.Conv2d, conv_kwargs=None, norm_op=nn.BatchNorm2d, 
                 norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        
        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs, conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels
        
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(*([basic_block(input_feature_channels, output_feature_channels, self.conv_op, self.conv_kwargs_first_conv, self.norm_op, 
                                                   self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs)] +
                                      [basic_block(output_feature_channels, output_feature_channels, self.conv_op, self.conv_kwargs, self.norm_op, 
                                                   self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))
    
    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class hwUpsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(hwUpsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000 # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2, feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, 
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None, 
                 highway = False, deep_supervision=False, anchor_out=False, dropout_in_localization=False, final_nonlin=sigmoid_helper, 
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None, upscale_logits=False, convolutional_pooling=False, 
                 convolutional_upsampling=False, max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same
        Does this look complicated? Nah bro. Functionality > usability
        This does everything you need, including world peace.
        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits # 该参数是唯二从UNetModel_newpreview中输入的参数, upscale_logits=True
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.anchor_out = anchor_out # 该参数是唯二从UNetModel_newpreview中输入的参数, anchor_out=True
        self.highway = highway

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.conv_trans_blocks_a = []
        self.conv_trans_blocks_b = []
        self.td = [] # to down sample
        self.tu = [] # to up sample
        self.ffparser = []
        self.seg_outputs = []

        output_features, input_features = base_num_features, input_channels
        
        # 创建编码器架构
        for d in range(num_pool): # num_pool从class UNetModel_newpreview输入, num_pool=5
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None
            
            # 添加卷积层参数: 核心尺寸、填充尺寸
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage, self.conv_op, self.conv_kwargs, 
                                                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, 
                                                              self.nonlin, self.nonlin_kwargs, first_stride, basic_block=basic_block))
            
            # 编码器除了最后卷积块, 均拼接以下卷积层
            if d < num_pool - 1 and self.highway:
                self.conv_trans_blocks_a.append(conv_nd(2, int(d / 2 + 1) * 128, 2 **(d + 5), 1))
                self.conv_trans_blocks_b.append(conv_nd(2, 2 **(d + 5), 1, 1))
            
            # 这两种写法完全一样, 为啥不放一起???
            if d != num_pool - 1 and self.highway:
                self.ffparser.append(FFParser(output_features, 256 // (2 **(d+1)), 256 // (2 **(d+2))+1)) # 这TM是MedSegDiff-V1的架构!!!
            
            # 该参数默认False，所以默认拼接池化层
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            
            input_features = output_features # 该步是为了赋予后续瓶颈层输入通道数
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)
        
        # now the bottleneck.
        # determine the first stride 该参数默认False
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using convolutional upsampling.
        # If we use convolutional upsampling then the reduction in feature maps will be done by the transposed conv
        if self.convolutional_upsampling: # 该参数默认False
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels
        
        # 添加卷积层参数: 核心尺寸、填充尺寸
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        
        # 创建瓶颈层架构
        self.conv_blocks_context.append(nn.Sequential(StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                                                                        self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                                                                        self.nonlin_kwargs, first_stride, basic_block=basic_block),
                                                      StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                                                        self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                                                                        self.nonlin_kwargs, basic_block=basic_block)))
        
        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0
        
        # 创建解码器架构 now lets build the localization pathway
        for u in range(num_pool): # num_pool=5
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip
            
            # 该参数默认为False
            if not self.convolutional_upsampling:
                self.tu.append(hwUpsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)], pool_op_kernel_sizes[-(u + 1)], bias=False))
            
            # 添加卷积层参数: 核心尺寸、填充尺寸
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            
            # add convolutions
            self.conv_blocks_localization.append(nn.Sequential(StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                                                                 self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                                                                 self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                                                               StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                                                                 self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                                                                 self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
        
        # 该参数默认为False
        if self._deep_supervision:
            for ds in range(len(self.conv_blocks_localization)):
                self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))
        else:
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))
        
        self.upscale_logits_ops, cum_upsample = [], np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        
        # logits是全连接层输出后、进入Softmax前的东西, 这个参数从UNetModel_newpreview输入, upscale_logits=True
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(hwUpsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]), mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)
        
        # dropout概率默认设置为0
        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_trans_blocks_a = nn.ModuleList(self.conv_trans_blocks_a)
        self.conv_trans_blocks_b = nn.ModuleList(self.conv_trans_blocks_b)
        self.ffparser = nn.ModuleList(self.ffparser)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits: self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops) # lambda x:x is not a Module so we need to distinguish here
        
        # 初始化网络权重和偏置常数
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)


    def forward(self, x, hs = None):
        skips, seg_outputs, anch_outputs = [], [], []
        
        # 编码器模块
        # 减一是因为还拼接了1个中间层
        for d in range(len(self.conv_blocks_context) - 1):
            # 逐卷积块处理特征层
            x = self.conv_blocks_context[d](x)
            # 将这些特征层收集起来用于侧连接, len(skips)=5
            skips.append(x)
            # 该参数为False, 所以not False=True, 此句会被执行
            if not self.convolutional_pooling:
                x = self.td[d](x)
            # 这个参数是False, UNetModel_newpreview并没有输入这个参数
            if hs:
                h = hs.pop(0)
                ddims = h.size(1)
                h = self.conv_trans_blocks_a[d](h)
                h = self.ffparser[d](h)
                ha = self.conv_trans_blocks_b[d](h)
                hb = th.mean(h,(2,3))
                hb = hb[:,:,None,None]
                x = x * ha * hb
        
        # 中间嵌入模块
        # 经过一次卷积处理后再生成嵌入
        x = self.conv_blocks_context[-1](x)
        emb = conv_nd(2, x.size(1), 512, 1).to(device = x.device)(x)
        
        # 解码器模块
        for u in range(len(self.tu)):
            # Upsampling.
            x = self.tu[u](x)
            # 再拼接侧连接特征层
            x = th.cat((x, skips[-(u + 1)]), dim=1)
            # 最后卷积块处理
            x = self.conv_blocks_localization[u](x)
            # _deep_supervision=False
            if self._deep_supervision:
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            # anchor_out为True, 再收集特征层
            if self.anchor_out and (not self._deep_supervision):
                anch_outputs.append(x)
        
        # 最终输出层 final_nonlin就是sigmoid
        # 当seg_outputs这个列表为空时, not seg_outputs=True, 此句会被执行
        if not seg_outputs:
            seg_outputs.append(self.final_nonlin(self.seg_outputs[0](x)))
        
        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        if self.anchor_out: # 经过测试代码实际上输出这行
            return tuple([i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], anch_outputs[:-1][::-1])]), seg_outputs[-1]
            # anch_outputs是个列表, 其中有5个元素, 即解码器特征层, 但只输出前4个; 并且以上操作将anch_outputs内前4个元素倒序输出
            # anch_outputs[:-1][::-1]输出的特征层Shape为[1, 32, 32, 32] [1, 64, 16, 16] [1, 128, 8, 8] [1, 256, 4, 4]
            # seg_outputs也是列表, 其中只有1个元素, 即最后输出的分割掩膜, Shape=[1, 1, 64, 64]
            # upscale_logits_ops中包含四个hwUpsample()函数, 用于将anch_outputs中的特征层上采样到相同尺寸, 即64×64
        else:
            return emb, seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features, num_modalities,
                                        num_classes, pool_op_kernel_sizes, deep_supervision=False, conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
        return tmp
