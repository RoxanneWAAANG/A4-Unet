import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

# Import models and utilities
from evaluate import evaluate
from utils.dice_score import dice_loss
from utils.data_loading import BasicDataset, CarvanaDataset
from a4unet.dataloader.bratsloader import BRATSDataset3D
from a4unet.dataloader.hippoloader import HIPPODataset3D
from a4unet.dataloader.isicloader import ISICDataset
from a4unet.a4unet import create_a4unet_model

# Directory paths for various datasets and logs
dir_brats = Path('./datasets/brats/2020')
dir_img = Path('./datasets/isic2018/train')
dir_mask = Path('./datasets/isic2018/masks')
dir_checkpoint = Path('./checkpoints')
dir_tensorboard = Path('./')

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Argument parser for command line arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the A4UNet model on images and target masks')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    return parser.parse_args()

# Function to train the model
def train_net(net, device, epochs=5, batch_size=1, learning_rate=0.001, val_percent=0.1, save_checkpoint=True, img_scale=0.5, amp=False):
    # Set up directories for saving checkpoints and logs
    dir_checkpoint = Path('./checkpoints/')
    dir_checkpoint.mkdir(parents=True, exist_ok=True)

    # Set up TensorBoard writer
    writer = SummaryWriter(comment=f'LR_{learning_rate}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    # Define optimizer and loss function
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()

    # Load datasets
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent / 100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # Set up data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # Training loop
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32 if net.n_classes == 1 else torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Validation after each epoch
            val_score = evaluate(net, val_loader, device, amp)
            scheduler.step(val_score)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            if net.n_classes > 1:
                logging.info('Validation cross entropy: {}'.format(val_score))
                writer.add_scalar('Loss/test', val_score, global_step)
            else:
                logging.info('Validation Dice Coeff: {}'.format(val_score))
                writer.add_scalar('Dice/test', val_score, global_step)

            # Save checkpoint
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch + 1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved!')

    writer.close()

# Main function to set up the training process
if __name__ == '__main__':
    args = get_args()

    # Set up device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Create the model
    net = create_a4unet_model(n_channels=3, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # Load model weights if specified
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    # Start training
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.learning_rate,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.validation,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)