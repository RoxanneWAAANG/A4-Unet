# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:14:38 2023

@author: Pilot Crysi
"""
import os
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from tqdm import tqdm
from torch import optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from unet import UNet
from evaluate import evaluate
from utils.dice_score import dice_loss
from utils.data_loading import BasicDataset, CarvanaDataset

from a4unet.dataloader.bratsloader import BRATSDataset3D
from a4unet.dataloader.hippoloader import HIPPODataset3D
from a4unet.dataloader.isicloader import ISICDataset
from a4unet.a4unet import create_a4unet_model
from a4unet.lr_scheduler import LinearWarmupCosineAnnealingLR

# dir_img = Path('/home/fyp1/ChengYuxuan/UNetMilesial-MedSegDiff/datasets/test_kfold_inputs/')
# dir_mask = Path('/home/fyp1/ChengYuxuan/UNetMilesial-MedSegDiff/datasets/test_kfold_masks/')

dir_brats = Path('/root/autodl-tmp/brats21_test/')

dir_checkpoint = Path('/root/autodl-tmp/A4-Unet/checkpoints/')
dir_tensorboard = Path('/root/tf-logs/')


def validation(model, device, batch_size: int = 1, save_checkpoint: bool = True, img_scale: float = 0.5, amp: bool = False, medsegdiff: bool = False,
               datasets: bool = False, input_size: int = 256, weight_decay: float = 1e-8, momentum: float = 0.999, gradient_clipping: float = 1.0):
    
    # 1. Create dataset
    try:
        if datasets != 'Brats':
            dataset = CarvanaDataset(dir_img, dir_mask, img_scale, medsegdiff, input_size)
        else:
            tran_list = [transforms.Resize((input_size, input_size))]
            transform_train = transforms.Compose(tran_list)
            dataset = BRATSDataset3D(dir_brats, transform_train, test_flag=False)
    except (AssertionError, RuntimeError, IndexError):
        if datasets != 'Brats':
            dataset = BasicDataset(dir_img, dir_mask, img_scale, medsegdiff, input_size)
        else:
            tran_list = [transforms.Resize((input_size, input_size))]
            transform_train = transforms.Compose(tran_list)
            dataset = BRATSDataset3D(dir_brats, transform_train, test_flag=False)

    # 3. Create data loaders
    loader_args_test = dict(batch_size=1, num_workers=0, pin_memory=True)
    val_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args_test)
    
    # Set up the loss scaling for AMP
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Evaluation round must outside batch level for-loop
    logging.info(f'''Starting validation''')
    
    # Ensure at least one parameter has requires_grad=True
    at_least_one_requires_grad = any(p.requires_grad for p in model.parameters())
    if not at_least_one_requires_grad:
        logging.warning("None of the model parameters have requires_grad=True. Setting requires_grad=True for the first parameter.")
        for param in model.parameters():
            param.requires_grad = True
            
    # evaluate validation dataset dice
    val_score = evaluate(model, val_loader, device, amp, datasets, False)
            
    # print & log validation dice
    logging.info('Validation Dice score: {}'.format(val_score[0]))
    logging.info('Validation mIoU score: {}'.format(val_score[1]))
    logging.info('Validation HD95 score: {}'.format(val_score[2]))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--batch-size',    '-b',  type=int,     default=1,     dest='batch_size', help='Batch size')
    parser.add_argument('--load',          '-f',  type=str,     default='/root/autodl-tmp/A4-Unet/checkpoints/checkpoint_epoch10.pth')
    parser.add_argument('--scale',         '-s',  type=float,   default=1.0,                      help='Images Downscaling factor')
    parser.add_argument('--amp',           action='store_true', default=False,                    help='Mixed Precision')
    parser.add_argument('--bilinear',      action='store_true', default=False,                    help='Bilinear upsampling')
    parser.add_argument('--classes',       '-c',  type=int,     default=2,                        help='Number of classes')
    parser.add_argument('--medsegdiff',    action='store_true', default=True,  dest='a4',       help='Enable MedSegDiff Arch')
    parser.add_argument('--datasets',      '-d', type=str,      default='Brats', dest='datasets', help='Choose Dataset')
    parser.add_argument('--input_size',    '-i',  type=int,     default=128,   dest='input_size', help='Input Size of MedSegDiff')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args() # 加载基础参数
    
    # 检测并设定设备
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}! Titan System Initiating!')
    
    if args.datasets: input_channel = 4
    else: input_channel = 3

    # 加载模型实例
    if not args.a4: # args.diff即是否启用MedSegDiff架构
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        model = model.to(memory_format=torch.channels_last) # 涉及到张量在显存中的储存特性, https://blog.csdn.net/hxxjxw/article/details/124209275
    else:
        model = create_a4unet_model(image_size=args.input_size, num_channels=128, num_res_blocks=2, num_classes=args.classes, learn_sigma=True, in_ch=input_channel)
    
    logging.info(f'Model loaded, Welcome Back, Pilot!')
    
    # 加载预训练权重
    state_dict = torch.load(args.load, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {args.load}')
    
    # 模型加载入设备
    model.to(device=device)
    
    # 启动训练
    try:
        validation(model=model, batch_size=args.batch_size, device=device, amp=args.amp, medsegdiff=args.a4, datasets=args.datasets, input_size=args.input_size)
    except torch.cuda.CudaError as e:
        print(f"CUDA out of memory error: {str(e)}")
        torch.cuda.empty_cache()
        model.use_checkpointing()
        validation(model=model, batch_size=args.batch_size, device=device, amp=args.amp, medsegdiff=args.a4, datasets=args.datasets, input_size=args.input_size)
