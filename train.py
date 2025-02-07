# !!! environment:
# conda activate unet_att

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

from evaluate import evaluate
from utils.dice_score import dice_loss
from utils.data_loading import BasicDataset, CarvanaDataset

from a4unet.dataloader.bratsloader import BRATSDataset3D
from a4unet.a4unet import create_a4unet_model
from model.unet import UNet

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")

# BraTS Dataset.
# dir_brats = Path('/root/autodl-tmp/train')
dir_brats = Path('C:/Users/admin/Desktop/Spring2025/AIPI540/CV/MICCAI_BraTS2020_TrainingData')
dir_img = Path('/')
dir_mask = Path('/')

dir_checkpoint = Path('checkpoints')
dir_tensorboard = Path('tf-logs')


def train_model(model, device, epochs: int = 20, batch_size: int = 16, learning_rate: float = 1e-5, val_percent: float = 0.5, val_step: float = 10, 
                save_checkpoint: bool = True, img_scale: float = 0.5, amp: bool = False, a4unet: bool = False, datasets: str = 'Brats', input_size: int = 256, 
                weight_decay: float = 1e-8, momentum: float = 0.999, gradient_clipping: float = 1.0):
    
    # 1. Create dataset
    try:
        if datasets == 'Brats':
            train_list = [transforms.Resize((input_size, input_size), antialias=True)]
            transform_train = transforms.Compose(train_list)
            # print("path: ", dir_brats)
            dataset = BRATSDataset3D(dir_brats, transform_train, test_flag=False)
        else:
            dataset = CarvanaDataset(dir_img, dir_mask, img_scale, a4unet, input_size)
    except (AssertionError, RuntimeError, IndexError):
        if datasets == 'Brats':
            train_list = [transforms.Resize((input_size, input_size), antialias=True)]
            transform_train = transforms.Compose(train_list)
            dataset = BRATSDataset3D(dir_brats, transform_train, test_flag=False)
        else:
            dataset = BasicDataset(dir_img, dir_mask, img_scale, a4unet, input_size)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args_train = dict(batch_size=batch_size, num_workers=10, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args_train)
    loader_args_test = dict(batch_size=1, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args_test)

    tblogger = SummaryWriter(os.path.join(dir_tensorboard, "tensorboard"))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    
    # 4. Set up the optimizer, the loss, the learning rate scheduler
    if not a4unet:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum) # foreach=True
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) # goal: maximize Dice score
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0) # lr=2e-4, weight_decay=1e-3
    
    # Set up the loss scaling for AMP
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0 # 初始化全局轮次数

    # 5. Begin training Epoch级循环
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        # Batch级循环
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # Fetch Image and GroundTruth
                if datasets == 'Brats':
                    # images, true_masks = batch['image'], batch['mask']
                    images, true_masks = batch[0], batch[1]
                else:
                    images, true_masks, name = batch
                    
                if datasets == 'Brats':
                    true_masks = torch.squeeze(true_masks, dim=1)
                elif datasets == 'ISIC':
                    # If true_masks has an extra dimension that's not needed:
                    true_masks = true_masks.squeeze(1)

                assert images.shape[1] == model.n_channels, f'Network has been defined with {model.n_channels} input channels, ' \
                                                            f'but loaded images have {images.shape[1]} channels. ' \
                                                             'Please check that the images are loaded correctly.'
                
                # 图片与GT加载入设备
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # 前向传播取得预测掩膜
                    masks_pred = model(images)
                    
                    # 计算损失函数, 更新网络
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(F.softmax(masks_pred, dim=1).float(), F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(), multiclass=True)
                
                # 更新优化器
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                # 更新进度条
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                tblogger.add_scalar("train/loss", loss.item(), epoch)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
        # Evaluation round must outside batch level for-loop
        if validation_step(epoch, val_step) == True: # be adviced epoch started at 1
            logging.info(f'''Starting validation''')
            
            # evaluate validation dataset dice
            val_score = evaluate(model, val_loader, device, amp, datasets, False)
            
            # update learning rate scheduler
            if not a4unet:
                scheduler.step(val_score[0])
            
            # print & log validation dice
            logging.info('Validation Dice score: {}'.format(val_score[0]))
            logging.info('Validation mIoU score: {}'.format(val_score[1]))

            tblogger.add_scalar("val/score", val_score[0], epoch)
            
            # save checkpoint with evaluation frequency
            # if save_checkpoint and epoch % 5 == 0:
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'sspp_checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')


def validation_step(epoch, val_step):
	if epoch % val_step == 0:
		return True


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs',        '-e',  type=int,     default=20,                      help='Number of epochs')
    parser.add_argument('--batch-size',    '-b',  type=int,     default=16,    dest='batch_size', help='Batch size')
    parser.add_argument('--learning-rate', '-l',  type=float,   default=1e-5,  dest='lr',         help='Learning rate')
    parser.add_argument('--load',          '-f',  type=str,     default=False,                    help='Load Pre-train model')
    parser.add_argument('--scale',         '-s',  type=float,   default=1.0,                      help='Images Downscaling factor')
    parser.add_argument('--validation',    '-v',  type=float,   default=10.0,  dest='val',        help='Percent of val data (0-100)')
    parser.add_argument('--valstep',       '-vs', type=float,   default=1.0,                      help='Validation Steps')
    parser.add_argument('--amp',           action='store_true', default=False,                    help='Mixed Precision')
    parser.add_argument('--bilinear',      action='store_true', default=False,                    help='Bilinear upsampling')
    parser.add_argument('--classes',       '-c',  type=int,     default=2,                        help='Number of classes')
    parser.add_argument('--a4unet',    action='store_true', default=False,  dest='a4',       help='Enable A4Unet Arch')
    parser.add_argument('--datasets',      '-d', type=str,      default='Brats', dest='datasets', help='Choose Dataset')
    parser.add_argument('--input_size',    '-i',  type=int,     default=128,   dest='input_size', help='Input Size of A4Unet')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args() # 加载基础参数
    
    # 检测并设定设备
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}! Titan System Initiating!')
    
    if args.datasets == 'Brats' or args.datasets == 'Hippo':
        input_channel = 4
    else:
        input_channel = 3

    # 加载模型实例
    # Change here to adapt to your data, n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if not args.a4: # args.diff默认为False
        print('Model U-Net is initiating!!!')
        model = UNet(n_channels=input_channel, n_classes=args.classes, bilinear=args.bilinear)
        # 涉及到张量在显存中的储存特性, https://blog.csdn.net/hxxjxw/article/details/124209275
        model = model.to(memory_format=torch.channels_last)
    else:
        print('Model A4-Unet is initiating!!!')
        model = create_a4unet_model(image_size=args.input_size, num_channels=128, num_res_blocks=2, num_classes=args.classes, learn_sigma=True, in_ch=input_channel)
    
    logging.info(f'Model loaded, Control transfer to Pilot!')
    
    # 加载预训练权重
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    
    # 模型加载入设备
    model.to(device=device)
    
    # 启动训练
    try:
        train_model(model=model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device, img_scale=args.scale, 
                    val_percent=args.val / 100, val_step=args.valstep, amp=args.amp, a4unet=args.a4, datasets=args.datasets, input_size=args.input_size)
    except torch.cuda.CudaError as e:
        print(f"CUDA out of memory error: {str(e)}")
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(model=model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device, img_scale=args.scale, 
                    val_percent=args.val / 100, val_step=args.valstep, amp=args.amp, a4unet=args.a4, datasets=args.datasets, input_size=args.input_size)
                    
