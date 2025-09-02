# -*- coding: utf-8 -*-

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

# Dataset paths - MODIFY THESE ACCORDING TO YOUR SETUP
dir_img = Path('./data/test_inputs/')
dir_mask = Path('./data/test_masks/')
dir_brats = Path('./data/brats21_test/')
dir_checkpoint = Path('./checkpoints/')
dir_tensorboard = Path('./logs/')


def validation(model, device, datasets, input_size):
    """
    Validation function for medical image segmentation models.
    
    Args:
        model: Neural network model to validate
        device: Device to run validation on (cuda/cpu)
        datasets: Dataset name ('Brats', 'ISIC', etc.)
        input_size: Input image size for resizing
    """
    
    # 1. Create dataset
    try:
        if datasets != 'Brats':
            dataset = CarvanaDataset(dir_img, dir_mask, 0.5, True, input_size)
        else:
            tran_list = [transforms.Resize((input_size, input_size), antialias=True)]
            transform_train = transforms.Compose(tran_list)
            dataset = BRATSDataset3D(dir_brats, transform_train, test_flag=False)
    except (AssertionError, RuntimeError, IndexError):
        if datasets != 'Brats':
            dataset = BasicDataset(dir_img, dir_mask, 0.5, True, input_size)
        else:
            tran_list = [transforms.Resize((input_size, input_size), antialias=True)]
            transform_train = transforms.Compose(tran_list)
            dataset = BRATSDataset3D(dir_brats, transform_train, test_flag=False)

    # 2. Create data loaders
    loader_args_test = dict(batch_size=1, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args_test)
    
    # 3. Start validation
    logging.info(f'''Starting validation on {datasets} dataset''')
    
    # Ensure at least one parameter has requires_grad=True
    at_least_one_requires_grad = any(p.requires_grad for p in model.parameters())
    if not at_least_one_requires_grad:
        logging.warning("Setting requires_grad=True for model parameters")
        for param in model.parameters():
            param.requires_grad = True
            break
            
    # 4. Run evaluation
    val_score = evaluate(model, val_loader, device, amp=False, dataset_name=datasets, final_test=False)
            
    # 5. Log results
    logging.info('Validation Dice score: {:.4f}'.format(val_score[0]))
    logging.info('Validation mIoU score: {:.4f}'.format(val_score[1]))
    logging.info('Validation HD95 score: {:.4f}'.format(val_score[2]))


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate segmentation models')
    parser.add_argument('--batch-size',    '-b',  type=int,     default=1,     dest='batch_size', help='Batch size')
    parser.add_argument('--load',          '-f',  type=str,     default='./checkpoints/checkpoint_epoch10.pth')  # Fixed path
    parser.add_argument('--scale',         '-s',  type=float,   default=1.0,                      help='Images Downscaling factor')
    parser.add_argument('--amp',           action='store_true', default=False,                    help='Mixed Precision')
    parser.add_argument('--bilinear',      action='store_true', default=False,                    help='Bilinear upsampling')
    parser.add_argument('--classes',       '-c',  type=int,     default=2,                        help='Number of classes')
    parser.add_argument('--medsegdiff',    action='store_true', default=True,  dest='a4',       help='Enable A4-UNet Architecture')
    parser.add_argument('--datasets',      '-d', type=str,      default='Brats', dest='datasets', help='Choose Dataset')
    parser.add_argument('--input_size',    '-i',  type=int,     default=128,   dest='input_size', help='Input Size')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # Setup logging and device
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}! System Initiating!')
    
    # Determine input channels based on dataset
    if args.datasets == 'Brats': 
        input_channel = 4  # T1, T1ce, T2, FLAIR
    else: 
        input_channel = 3  # RGB
    
    # Create model
    if not args.a4:  # Standard UNet
        model = UNet(n_channels=input_channel, n_classes=args.classes, bilinear=args.bilinear)
        model = model.to(memory_format=torch.channels_last)
    else:  # A4-UNet architecture
        model = create_a4unet_model(
            image_size=args.input_size, 
            num_channels=128, 
            num_res_blocks=2, 
            num_classes=args.classes, 
            learn_sigma=True, 
            in_ch=input_channel
        )
    
    logging.info(f'Model created: {model.__class__.__name__}')
    
    # Load pretrained weights
    try:
        state_dict = torch.load(args.load, map_location=device, weights_only=True)
        # Remove mask_values if present
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    except Exception as e:
        logging.error(f'Failed to load checkpoint: {e}')
        exit(1)
    
    # Move model to device
    model.to(device=device)
    
    # Run validation
    try:
        validation(
            model=model, 
            device=device, 
            datasets=args.datasets, 
            input_size=args.input_size
        )
    except torch.cuda.CudaError as e:
        logging.error(f"CUDA out of memory: {str(e)}")
        torch.cuda.empty_cache()
        # Retry with checkpointing if available
        if hasattr(model, 'use_checkpointing'):
            model.use_checkpointing()
        validation(
            model=model, 
            device=device, 
            datasets=args.datasets, 
            input_size=args.input_size
        )
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        exit(1)
    
    logging.info("Validation completed successfully!")