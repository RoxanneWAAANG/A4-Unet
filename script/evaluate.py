
import os
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import directed_hausdorff


out_files = './env_result'

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

def evaluate(net, dataloader, device, amp, datasets, final_test):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    mIoU = 0
    hd95 = 0

    if final_test:
        save_pred = True
    else:
        save_pred = False

    # iterate over the validation set
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            if datasets == 'Brats':
                image, mask_true = batch[0], batch[1]
            else:
                image, mask_true, name = batch
                    
            if datasets == 'Brats':
                mask_true = torch.squeeze(mask_true, dim=1)
            elif datasets == 'ISIC':
                mask_true = mask_true.squeeze(1)

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)
            
            if save_pred:
                out_filename = out_files + name[0].split('/')[-2] + '_' + name[0].split('/')[-1][0:-4] + '.jpg'
                maskpred = F.interpolate(mask_pred, (image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
                maskpred = maskpred.argmax(dim=1)
                maskpred = maskpred[0].cpu().long().squeeze().numpy()
                result = mask_to_image(maskpred, [0, 1])
                result.save(out_filename)
            
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'

                if datasets == 'Brats':
                    mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                elif datasets == 'ISIC':
                    mask_pred = mask_pred.argmax(dim=1)
                
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                mIoU += jaccard_score(mask_true.argmax(dim=1).cpu().numpy().flatten(),
                                       mask_pred.argmax(dim=1).cpu().numpy().flatten(), average='micro')                
                hd95 += np.percentile(directed_hausdorff(mask_pred.cpu().numpy().reshape(1, -1), mask_true.cpu().numpy().reshape(1, -1))[0], 95)
                

    net.train()
    
    dice_score /= max(num_val_batches, 1)
    mIoU /= max(num_val_batches, 1)
    hd95 /= max(num_val_batches, 1)

    return dice_score, mIoU, hd95


