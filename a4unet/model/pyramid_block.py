# https://f.daixianiu.cn/csdn/49944081491655545.html

import numpy as np
import torch
import torch.nn as nn


class DilatedSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedSpatialPyramidPooling, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, dilation=4)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, dilation=8)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, dilation=16)

    def forward(self, x):
        x = self.bn(x)
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        c = torch.cat((d1, d2, d3, d4), dim=1)
        return c

