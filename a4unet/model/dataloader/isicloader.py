import os
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image
from skimage import io
from skimage.transform import rotate
from torch.utils.data.dataset import Dataset


class ISICDataset(Dataset):
    def __init__(self, img_path, mask_path, transform = None):
        self.img_path    = img_path
        self.mask_path   = mask_path
        self.transform   = transform
        self.num_classes = 2

        # 获取文件夹中所有图像文件的名称（假设图像和掩膜文件名一一对应）
        img_files = sorted(os.listdir(self.img_path))
        mask_files = sorted(os.listdir(self.mask_path))

        # 构建annotation_lines列表
        annotation_lines = []
        for img_file, mask_file in zip(img_files, mask_files):
            img_name = os.path.splitext(img_file)[0]  # 获取图像文件名，去除扩展名
            mask_name = os.path.splitext(mask_file)[0]  # 获取掩膜文件名，去除扩展名
            annotation_line = f"{img_name} {mask_name}"
            annotation_lines.append(annotation_line)

        self.annotation_lines = annotation_lines
        self.length      = len(annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Take the image name
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]
        # Take the image name
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # Construct image and mask paths
        img_path = os.path.join(self.img_path, f"{name}.jpg")
        mask_path = os.path.join(self.mask_path, f"{name}_segmentation.png")
        # img_path = os.path.join(self.img_path, f"ImageTr/{name}.jpg")
        # mask_path = os.path.join(self.mask_path, f"MaskTr/{name}_segmentation.png")

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return img, mask, name
        
        # # 进行标准图像处理
        # if self.transform: # 经测试会执行以下操作
        #     state = torch.get_rng_state()
        #     img = self.transform(img)
        #     torch.set_rng_state(state)
        #     mask = self.transform(mask)
        
        # # 构造二值化Mask, 用于构造[h, w, 3]形态张量
        # bimask = mask.squeeze(0) # mask.shape=[1, h, w], bimask.shape=[h, w]
        # bimask = np.array(bimask) # bimask为tensor, 需要转换成numpy, 否则后续会报错
        # bimask = (bimask * 255).astype(np.uint8) # 从float32转换成uint8
        # bimask[bimask >= self.num_classes] = self.num_classes # 前景则赋1, 背景则赋0
        
        # # 构造[h, w, 3]形态张量, 用于进行dice loss计算
        # seg_labels  = np.eye(self.num_classes + 1)[bimask.reshape([-1])]
        # seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        
        # return img, mask, seg_labels