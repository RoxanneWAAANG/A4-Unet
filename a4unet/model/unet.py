""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn

from .unet_parts import *
from .utils import *
from .sspp import SwinASPP


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        #################### ASPP ##################
        # Feature fusion.
        # Add a feature pyramid layer
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
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),  # Removed bias since BatchNorm is added
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        ############################################

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        ################ ASPP ##################
        # Feature fusion -- ASPP.
        x5 = x5.permute(0, 2, 3, 1)
        x = self.aspp(x5)
        x = x.permute(0, 3, 1, 2)
        x = self.bottleneck(x)
        ########################################
        # print(x.shape)#: torch.Size([16, 512, 8, 8])
        # print(x4.shape)#: torch.Size([16, 512, 16, 16])

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
