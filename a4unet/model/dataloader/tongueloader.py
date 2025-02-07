import os
import torch
import pickle
import nibabel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from skimage.transform import rotate

class TongueDataset(Dataset):
    def __init__(self, data_path, transform = None, test_flag = False, plane = False):
        # read in main data path
        self.data_path  = str(data_path)
        self.train_img = os.path.join(self.data_path + 'train_img/')
        self.test_img = os.path.join(self.data_path + 'test_img/')
        
        self.transform = transform
        self.test_flag = test_flag

    def __len__(self):
        return len(self.train_img)

    def __getitem__(self, index):
        # Get the images and masks
        if self.test_flag == False:
            print("in train")
            print(self.train_img[index])
            print()
            nib_img = nibabel.load(self.data_path + '/train_img/' + self.train_img[index])
            nib_gt = nibabel.load(self.data_path + '/train_gt/' + self.train_img[index][0:-7] + '_gt.nii.gz')
            img, lab = nib_img.get_fdata(), nib_gt.get_fdata()
        else:
            print("in test")
            print(index)
            print()
            nib_img = nibabel.load(self.data_path + '/test_img/' + self.test_img[index])
            nib_gt = nibabel.load(self.data_path + '/test_gt/' + self.test_img[index][0:-7] + '_gt.nii.gz')
            img, lab = nib_img.get_fdata(), nib_gt.get_fdata()
        	
        assert img.shape == lab.shape, 'the shape of image and label are not match, plz check dataset'
        
        img, lab = torch.tensor(img), torch.tensor(lab) # torch.where需要张量才能进行条件判断
        mask = torch.where(lab > 0, 1, 0).float() # merge all tumor classes into one
	
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask) # num_classes + 1 background , object
        
        # 3D to 2D, channels as batch, shape [c, h, w] -> [c, 1, h, w] batch size
        assert img.shape == mask.shape, 'the shape of image and label are not match, plz check dataset'
        img, mask = torch.unsqueeze(img, dim=1), torch.unsqueeze(mask, dim=1) # 给中间增加1个维度

        return (img, mask, self.data_path) # the path won't be used
        
