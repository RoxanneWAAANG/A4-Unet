# !!! environment:

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

# =============================================================================
# DATASET PATHS - MODIFY THESE ACCORDING TO YOUR SETUP
# =============================================================================
dir_brats = Path('./data/MICCAI_BraTS2020_TrainingData')  # BraTS dataset path
dir_img = Path('./')  # Generic image directory
dir_mask = Path('./')  # Generic mask directory

# Output directories
dir_checkpoint = Path('checkpoints')  # Model checkpoint save directory
dir_tensorboard = Path('tf-logs')  # TensorBoard log directory


def train_model(model, device, epochs: int = 20, batch_size: int = 16, learning_rate: float = 1e-5, 
                val_percent: float = 0.5, val_step: float = 10, save_checkpoint: bool = True, 
                img_scale: float = 0.5, amp: bool = False, a4unet: bool = False, datasets: str = 'Brats', 
                input_size: int = 256, weight_decay: float = 1e-8, momentum: float = 0.999, 
                gradient_clipping: float = 1.0):
    """
    Main training function for medical image segmentation models.
    
    Args:
        model: Neural network model (UNet or A4-UNet)
        device: Training device (cuda/cpu)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        val_percent: Percentage of data for validation (0-1)
        val_step: Validation frequency (every N epochs)
        save_checkpoint: Whether to save model checkpoints
        img_scale: Image scaling factor
        amp: Use automatic mixed precision
        a4unet: Whether using A4-UNet architecture
        datasets: Dataset name ('Brats', 'ISIC', etc.)
        input_size: Input image size for resizing
        weight_decay: Weight decay for optimizer
        momentum: Momentum for RMSprop optimizer
        gradient_clipping: Gradient clipping threshold
    """
    
    # =============================================================================
    # 1. DATASET CREATION AND LOADING
    # =============================================================================
    try:
        if datasets == 'Brats':
            # BraTS dataset: 4D medical images (T1, T1ce, T2, FLAIR)
            train_list = [transforms.Resize((input_size, input_size), antialias=True)]
            transform_train = transforms.Compose(train_list)
            dataset = BRATSDataset3D(dir_brats, transform_train, test_flag=False)
        else:
            # Other datasets (Carvana, ISIC, etc.)
            dataset = CarvanaDataset(dir_img, dir_mask, img_scale, a4unet, input_size)
    except (AssertionError, RuntimeError, IndexError):
        # Fallback dataset creation if first attempt fails
        if datasets == 'Brats':
            train_list = [transforms.Resize((input_size, input_size), antialias=True)]
            transform_train = transforms.Compose(train_list)
            dataset = BRATSDataset3D(dir_brats, transform_train, test_flag=False)
        else:
            dataset = BasicDataset(dir_img, dir_mask, img_scale, a4unet, input_size)

    # =============================================================================
    # 2. TRAIN/VALIDATION SPLIT
    # =============================================================================
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # =============================================================================
    # 3. DATA LOADERS SETUP
    # =============================================================================
    # Training data loader: shuffle=True, larger batch size
    loader_args_train = dict(batch_size=batch_size, num_workers=10, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args_train)
    
    # Validation data loader: shuffle=False, batch_size=1
    loader_args_test = dict(batch_size=1, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args_test)

    # TensorBoard logger setup
    tblogger = SummaryWriter(os.path.join(dir_tensorboard, "tensorboard"))

    # Training information logging
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
    
    # =============================================================================
    # 4. OPTIMIZER, LOSS, SCHEDULER SETUP
    # =============================================================================
    if not a4unet:
        # Standard UNet: RMSprop optimizer with ReduceLROnPlateau scheduler
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # Maximize Dice score
    else:
        # A4-UNet: AdamW optimizer (no scheduler)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    
    # Mixed precision training setup
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # Loss function: CrossEntropy for multi-class, BCE for binary
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0  # Global step counter for logging

    # =============================================================================
    # 5. TRAINING LOOP - EPOCH LEVEL
    # =============================================================================
    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        epoch_loss = 0
        
        # Progress bar for current epoch
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            # =============================================================================
            # TRAINING LOOP - BATCH LEVEL
            # =============================================================================
            for batch in train_loader:
                # Unpack batch data based on dataset type
                if datasets == 'Brats':
                    images, true_masks = batch[0], batch[1]
                else:
                    images, true_masks, name = batch
                    
                # Dataset-specific mask preprocessing
                if datasets == 'Brats':
                    true_masks = torch.squeeze(true_masks, dim=1)
                elif datasets == 'ISIC':
                    true_masks = true_masks.squeeze(1)

                # Validate input channels match model expectations
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. ' \
                    'Please check that the images are loaded correctly.'
                
                # Move data to device with optimized memory format
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Forward pass with automatic mixed precision
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # Get model predictions
                    masks_pred = model(images)
                    
                    # Calculate loss based on number of classes
                    if model.n_classes == 1:
                        # Binary segmentation: BCE + Dice loss
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # Multi-class segmentation: CrossEntropy + Dice loss
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                
                # Backward pass and optimization
                optimizer.zero_grad(set_to_none=True)  # Clear gradients
                grad_scaler.scale(loss).backward()  # Backward pass with gradient scaling
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # Gradient clipping
                grad_scaler.step(optimizer)  # Optimizer step
                grad_scaler.update()  # Update gradient scaler
                
                # Progress tracking and logging
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                tblogger.add_scalar("train/loss", loss.item(), epoch)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
        # =============================================================================
        # VALIDATION PHASE
        # =============================================================================
        if validation_step(epoch, val_step) == True:  # Check if validation should run
            logging.info(f'''Starting validation''')
            
            # Run evaluation on validation set
            val_score = evaluate(model, val_loader, device, amp, datasets, False)
            
            # Update learning rate scheduler (only for standard UNet)
            if not a4unet:
                scheduler.step(val_score[0])  # Step based on Dice score
            
            # Log validation results
            logging.info('Validation Dice score: {}'.format(val_score[0]))
            logging.info('Validation mIoU score: {}'.format(val_score[1]))

            # TensorBoard logging
            tblogger.add_scalar("val/score", val_score[0], epoch)
            
            # =============================================================================
            # CHECKPOINT SAVING
            # =============================================================================
            if save_checkpoint:
                # Create checkpoint directory if it doesn't exist
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                
                # Save model state dict with mask values
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'sspp_checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')


def validation_step(epoch, val_step):
    """
    Determine if validation should run at current epoch.
    
    Args:
        epoch (int): Current epoch number
        val_step (float): Validation frequency
        
    Returns:
        bool: True if validation should run
    """
    if epoch % val_step == 0:
        return True


def get_args():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    # Training hyperparameters
    parser.add_argument('--epochs',        '-e',  type=int,     default=20,                      help='Number of epochs')
    parser.add_argument('--batch-size',    '-b',  type=int,     default=16,    dest='batch_size', help='Batch size')
    parser.add_argument('--learning-rate', '-l',  type=float,   default=1e-5,  dest='lr',         help='Learning rate')
    
    # Model loading and saving
    parser.add_argument('--load',          '-f',  type=str,     default=False,                    help='Load Pre-train model')
    parser.add_argument('--scale',         '-s',  type=float,   default=1.0,                      help='Images Downscaling factor')
    
    # Validation parameters
    parser.add_argument('--validation',    '-v',  type=float,   default=10.0,  dest='val',        help='Percent of val data (0-100)')
    parser.add_argument('--valstep',       '-vs', type=float,   default=1.0,                      help='Validation Steps')
    
    # Model architecture options
    parser.add_argument('--amp',           action='store_true', default=False,                    help='Mixed Precision')
    parser.add_argument('--bilinear',      action='store_true', default=False,                    help='Bilinear upsampling')
    parser.add_argument('--classes',       '-c',  type=int,     default=2,                        help='Number of classes')
    parser.add_argument('--a4unet',        action='store_true', default=False,  dest='a4',       help='Enable A4Unet Architecture')
    
    # Dataset configuration
    parser.add_argument('--datasets',      '-d', type=str,      default='Brats', dest='datasets', help='Choose Dataset')
    parser.add_argument('--input_size',    '-i',  type=int,     default=128,   dest='input_size', help='Input Size of A4Unet')

    return parser.parse_args()


if __name__ == '__main__':
    # =============================================================================
    # MAIN EXECUTION BLOCK
    # =============================================================================
    
    # Parse command line arguments
    args = get_args()
    
    # Setup logging and device detection
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}! Titan System Initiating!')
    
    # Determine input channels based on dataset type
    if args.datasets == 'Brats' or args.datasets == 'Hippo':
        input_channel = 4  # Medical images: T1, T1ce, T2, FLAIR
    else:
        input_channel = 3  # Natural images: RGB
    
    # =============================================================================
    # MODEL INITIALIZATION
    # =============================================================================
    if not args.a4:  # Standard UNet
        print('Model U-Net is initiating!!!')
        model = UNet(n_channels=input_channel, n_classes=args.classes, bilinear=args.bilinear)
        # Optimize tensor storage for better performance
        model = model.to(memory_format=torch.channels_last)
    else:  # A4-UNet architecture
        print('Model A4-Unet is initiating!!!')
        model = create_a4unet_model(
            image_size=args.input_size, 
            num_channels=128, 
            num_res_blocks=2, 
            num_classes=args.classes, 
            learn_sigma=True, 
            in_ch=input_channel
        )
    
    logging.info(f'Model loaded, Control transfer to Pilot!')
    
    # =============================================================================
    # PRETRAINED MODEL LOADING
    # =============================================================================
    if args.load:
        try:
            state_dict = torch.load(args.load, map_location=device)
            # Remove mask_values key if present (not part of model parameters)
            if 'mask_values' in state_dict:
                del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {args.load}')
        except Exception as e:
            logging.error(f'Failed to load pretrained model: {e}')
    
    # Move model to training device
    model.to(device=device)
    
    # =============================================================================
    # TRAINING EXECUTION
    # =============================================================================
    try:
        # Start training with all configured parameters
        train_model(
            model=model, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.lr, 
            device=device, 
            img_scale=args.scale, 
            val_percent=args.val / 100, 
            val_step=args.valstep, 
            amp=args.amp, 
            a4unet=args.a4, 
            datasets=args.datasets, 
            input_size=args.input_size
        )
    except torch.cuda.CudaError as e:
        # Handle CUDA out of memory errors
        print(f"CUDA out of memory error: {str(e)}")
        torch.cuda.empty_cache()  # Clear GPU memory cache
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'use_checkpointing'):
            model.use_checkpointing()
        
        # Retry training with memory optimizations
        train_model(
            model=model, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.lr, 
            device=device, 
            img_scale=args.scale, 
            val_percent=args.val / 100, 
            val_step=args.valstep, 
            amp=args.amp, 
            a4unet=args.a4, 
            datasets=args.datasets, 
            input_size=args.input_size
        )