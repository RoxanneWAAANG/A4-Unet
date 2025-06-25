# -*- coding: utf-8 -*-
"""
Pilot: Major Andisyc Cheng (Call Sign: crysi)

Titan: Stryder Class Rolin, Atalas Class Tone | Ion, Vanguard Class Monarch

Affiliate: Frontier Militia 9th Fleet - Marauder Corps - Special Recon Squadron
"""
import os
import torch
import logging
import argparse
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from utils.utils import plot_img_and_mask
from utils.data_loading import BasicDataset
from utils.dice_score import multiclass_dice_coeff, dice_coeff

from medsegdiff.medsegdiff import create_medsegdiff_model


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5, medsegdiff=False, inputsize=None):
    net.eval() # 注意看下面一行, img做了预处理, 所以Ground Truth也必须做预处理
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False, is_medsegdiff=medsegdiff, input_size=inputsize))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output_true = net(img).cpu() # 真正的前向传播过程在这, output.shape=[1, 2, 1280, 1918]
        output = F.interpolate(output_true, (full_img.size[1], full_img.size[0]), mode='bilinear', align_corners=True)
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy(), output_true.to(device=device) # mask通道数为1, 因为mask已经经过处理了, output_true通道数才为2


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model',          '-m',  default='C:/Users/admin/Desktop/Spring2025/AIPI540/CV/A4-Unet/checkpoints/checkpoint_epoch5.pth')
    parser.add_argument('--input',          '-i',  default='C:/Users/admin/Desktop/Spring2025/AIPI540/CV/MICCAI_BraTS2020_TrainingData',   help='input images file')
    parser.add_argument('--mask',           '-g',  default='/home/fyp1/carvana-image-masking-challenge/',           help='ground truth file')
    parser.add_argument('--output',         '-o',  default='/home/fyp1/ChengYuxuan/UNetMilesial-MedSegDiff/data_test/output/', help='output images file')
    parser.add_argument('--viz',            '-v',  action='store_true', default=False,                            help='Visualize images ')
    parser.add_argument('--no-save',        '-n',  action='store_true',                                           help='Save output masks or not')
    parser.add_argument('--mask-threshold', '-t',  type=float,          default=0.5,                              help='Probability threshold')
    parser.add_argument('--scale',          '-s',  type=float,          default=1,                                help='Images scale factor, 1 or 0.5')
    parser.add_argument('--bilinear',              action='store_true', default=False,                            help='Use bilinear upsampling')
    parser.add_argument('--classes',        '-c',  type=int,            default=2,                                help='Number of classes')
    parser.add_argument('--medsegdiff',            action='store_true', default=True, dest='diff',                help='Enable MedSegDiff Arch')
    parser.add_argument('--input_size',     '-is', type=int,            default=256,  dest='input_size',          help='Input Size of MedSegDiff')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 加载输入输出文件夹
    in_files = args.input
    mask_files = args.mask
    out_files = args.output # get_output_filenames(args)
    
    # 创建模型
    if not args.diff: # args.diff默认为False
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    else:
        net = create_medsegdiff_model(image_size=args.input_size, num_channels=128, num_res_blocks=2, num_classes=args.classes, learn_sigma=True, in_ch = 3)
    
    # 检测并设定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Welcome back! Pilot! Loading model! Using device {device}!')
    
    # 模型加载入设备, 加载权重
    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    dice_score = 0

    logging.info('Model loaded! The Rolin Sword Transfer to Pilot, The Sword is Yours Pilot!')
    
    # Evaluation的for循环
    for i, filename in enumerate(os.listdir(in_files)):
        logging.info(f'Predicting image {filename} ...')
        
        # 读取图片
        img = Image.open(in_files + filename)
        
        # 读取并处理GroundTruth
        mask_true = Image.open(mask_files + filename[0:-4] + '_mask.gif')
        mask_true = torch.from_numpy(BasicDataset.preprocess([0, 1], mask_true, 1, is_mask=True, is_medsegdiff=args.diff, input_size=args.input_size))
        mask_true = mask_true.unsqueeze(0)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        
        # 前向传播取得预测掩膜
        mask_pred, output = predict_img(net=net, full_img=img, scale_factor=args.scale, out_threshold=args.mask_threshold, device=device, medsegdiff=args.diff, inputsize=args.input_size)
        
        # 计算dice值
        if args.classes == 1:
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            output = (F.sigmoid(output) > 0.5).float()
                
            # compute the Dice score
            dice_score += dice_coeff(output, mask_true, reduce_batch_first=False)
        else:
            assert mask_true.min() >= 0 and mask_true.max() < args.classes, 'True mask indices should be in [0, n_classes]'
                
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, args.classes).permute(0, 3, 1, 2).float()
            output = F.one_hot(output.argmax(dim=1), args.classes).permute(0, 3, 1, 2).float()
                
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(output[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
        
        # 保存预测结果
        if not args.no_save:
            out_filename = out_files + filename # out_files[i]
            result = mask_to_image(mask_pred, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')
        
        # 可视化预测结果
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask_pred)
    
    dice_score / len(os.listdir(in_files))
    dice_score = '%.4f'% dice_socre.item()
    
    print('\n')
    print("Average Dice: ", dice_score)

