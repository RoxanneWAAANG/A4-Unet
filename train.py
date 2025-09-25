# !!! environment:

import os
import torch
from loguru import logger as _base_logger
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import json
from datetime import datetime

from tqdm import tqdm
from torch import optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from evaluate import evaluate
from a4unet.utils.dice_metrics import dice_loss
from a4unet.utils.data_loading import BasicDataset, CarvanaDataset

from a4unet.dataloader.bratsloader import BRATSDataset3D
from a4unet.model.a4unet import create_a4unet_model
from a4unet.model.unet import UNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")

# =============================================================================
# DEFAULT PATH PLACEHOLDERS (overridden by CLI)
# =============================================================================
dir_img = Path('./')
dir_mask = Path('./')

# Contextual logger for this module
log = _base_logger.bind(module="train")


def train_model(model, device, epochs: int = 20, batch_size: int = 16, learning_rate: float = 1e-5,
                val_percent: float = 0.5, val_step: float = 10, save_checkpoint: bool = True,
                img_scale: float = 0.5, amp: bool = False, a4unet: bool = False, datasets: str = 'Brats',
                input_size: int = 256, weight_decay: float = 1e-8, momentum: float = 0.999,
                gradient_clipping: float = 1.0,
                data_root: Path = Path('./'), outdir: Path = Path('./outputs'), run_name: str = 'run',
                num_workers: int = 4, save_best_only: bool = False, seed: int = 0,
                grad_accum_steps: int = 1, early_stop_patience: int = 0,
                scheduler_type: str = 'none', resume: str = '', save_every: int = 0):
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
    # 0. REPRODUCIBILITY & OUTPUT DIRS
    # =============================================================================
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    run_dir = Path(outdir) / run_name
    ckpt_dir = run_dir / 'checkpoints'
    tb_dir = run_dir / 'tb'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # save config
    try:
        config_path = run_dir / 'config.json'
        # do best effort to serialize
        config_dict = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'val_ratio': val_percent,
            'val_step': val_step,
            'save_checkpoint': save_checkpoint,
            'img_scale': img_scale,
            'amp': amp,
            'a4unet': a4unet,
            'datasets': datasets,
            'input_size': input_size,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'gradient_clipping': gradient_clipping,
            'data_root': str(data_root),
            'outdir': str(outdir),
            'run_name': run_name,
            'num_workers': num_workers,
            'save_best_only': save_best_only,
            'seed': seed,
            'grad_accum_steps': grad_accum_steps,
            'early_stop_patience': early_stop_patience,
            'scheduler': scheduler_type,
            'resume': resume,
            'save_every': save_every,
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    except Exception as e:
        log.warning(f'Failed to save config.json: {e}')

    # =============================================================================
    # 1. DATASET CREATION AND LOADING
    # =============================================================================
    try:
        if datasets == 'Brats':
            # BraTS dataset: 4D medical images (T1, T1ce, T2, FLAIR)
            train_list = [transforms.Resize((input_size, input_size), antialias=True)]
            transform_train = transforms.Compose(train_list)
            dataset = BRATSDataset3D(Path(data_root), transform_train, test_flag=False)
        else:
            # Other datasets (Carvana, ISIC, etc.)
            dataset = CarvanaDataset(dir_img, dir_mask, img_scale, a4unet, input_size)
    except (AssertionError, RuntimeError, IndexError):
        # Fallback dataset creation if first attempt fails
        if datasets == 'Brats':
            train_list = [transforms.Resize((input_size, input_size), antialias=True)]
            transform_train = transforms.Compose(train_list)
            dataset = BRATSDataset3D(Path(data_root), transform_train, test_flag=False)
        else:
            dataset = BasicDataset(dir_img, dir_mask, img_scale, a4unet, input_size)

    # =============================================================================
    # 2. TRAIN/VALIDATION SPLIT
    # =============================================================================
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # =============================================================================
    # 3. DATA LOADERS SETUP
    # =============================================================================
    # Training data loader: shuffle=True, larger batch size
    loader_args_train = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args_train)
    
    # Validation data loader: shuffle=False, batch_size=1
    loader_args_test = dict(batch_size=1, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args_test)

    # TensorBoard logger setup
    tblogger = SummaryWriter(str(tb_dir))

    # Training information logging
    log.info(f'''Starting training:
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    
    # Mixed precision training setup
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # Loss function: CrossEntropy for multi-class, BCE for binary
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0  # Global step counter for logging

    # =============================================================================
    # 5. TRAINING LOOP - EPOCH LEVEL
    # =============================================================================
    best_dice = float('-inf')
    epochs_no_improve = 0

    # Resume weights if provided
    if resume:
        try:
            state_dict = torch.load(resume, map_location=device)
            if 'mask_values' in state_dict:
                del state_dict['mask_values']
            model.load_state_dict(state_dict)
            log.info(f'Resumed model weights from {resume}')
        except Exception as e:
            log.error(f'Failed to resume from {resume}: {e}')

    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        epoch_loss = 0
        accum_counter = 0
        
        # Progress bar for current epoch
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            # =============================================================================
            # TRAINING LOOP - BATCH LEVEL
            # =============================================================================
            for batch_idx, batch in enumerate(train_loader):
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
                with torch.autocast(device.type, enabled=amp):
                    # Get model predictions
                    masks_pred = model(images)
                    
                    # Calculate loss based on number of classes
                    if model.n_classes == 1:
                        # Binary segmentation: BCE + Dice loss
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # Multi-class segmentation: CrossEntropy + Dice loss
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                
                # Backward pass and optimization with gradient accumulation
                loss_to_backprop = loss / max(1, grad_accum_steps)
                grad_scaler.scale(loss_to_backprop).backward()
                accum_counter += 1

                do_step = (accum_counter % grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
                if do_step:
                    # Unscale before clipping
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                # Progress tracking and logging
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                tblogger.add_scalar("train/loss", loss.item(), global_step)
                # Log learning rate
                current_lr = optimizer.param_groups[0]['lr']
                tblogger.add_scalar("train/lr", current_lr, global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
        # =============================================================================
        # VALIDATION PHASE
        # =============================================================================
        if validation_step(epoch, val_step) == True:  # Check if validation should run
            log.info(f'''Starting validation''')
            
            # Run evaluation on validation set
            val_score = evaluate(model, val_loader, device, amp, datasets, False)
            
            # Update learning rate scheduler
            if scheduler_type == 'plateau' and scheduler is not None:
                scheduler.step(val_score[0])  # Step based on Dice score
            
            # Log validation results
            log.info('Validation Dice score: {}'.format(val_score[0]))
            log.info('Validation mIoU score: {}'.format(val_score[1]))

            # TensorBoard logging
            tblogger.add_scalar("val/dice", val_score[0], epoch)
            tblogger.add_scalar("val/miou", val_score[1], epoch)
            
            # =============================================================================
            # CHECKPOINT SAVING
            # =============================================================================
            if save_checkpoint:
                improved = val_score[0] > best_dice
                if improved:
                    best_dice = val_score[0]
                    epochs_no_improve = 0
                    # Save best
                    state_dict = model.state_dict()
                    state_dict['mask_values'] = dataset.mask_values
                    torch.save(state_dict, str(ckpt_dir / 'best.pth'))
                    log.info(f'New best Dice {best_dice:.6f}; saved best.pth')
                else:
                    epochs_no_improve += 1

                # Optional periodic saving of last
                if (not save_best_only) and (save_every > 0) and (epoch % save_every == 0):
                    state_dict = model.state_dict()
                    state_dict['mask_values'] = dataset.mask_values
                    torch.save(state_dict, str(ckpt_dir / f'last_epoch{epoch:02d}.pth'))
                    log.info(f'Checkpoint last_epoch{epoch:02d}.pth saved!')

                # Early stopping
                if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                    log.info(f'Early stopping triggered after {epochs_no_improve} epochs without improvement.')
                    break

        # Step cosine scheduler per epoch
        if scheduler_type == 'cosine' and scheduler is not None:
            scheduler.step()


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
    parser = argparse.ArgumentParser(description='Train the UNet/A4-UNet on images and target masks')
    
    # Training hyperparameters
    parser.add_argument('--epochs',        '-e',  type=int,     default=1,                      help='Number of epochs')
    parser.add_argument('--batch-size',    '-b',  type=int,     default=16,    dest='batch_size', help='Batch size')
    parser.add_argument('--learning-rate', '-l',  type=float,   default=1e-4,  dest='lr',         help='Learning rate')
    
    # Model loading and saving
    parser.add_argument('--load',          '-f',  type=str,     default=False,                    help='Load Pre-train model')
    parser.add_argument('--resume',              type=str,      default='',                        help='Resume training from checkpoint (model weights)')
    parser.add_argument('--scale',         '-s',  type=float,   default=1.0,                      help='Images Downscaling factor')
    
    # Validation parameters
    parser.add_argument('--val-ratio',     '-vr', type=float,   default=0.1,   dest='val',        help='Validation ratio (0-1)')
    parser.add_argument('--valstep',       '-vs', type=float,   default=1.0,                      help='Validation Steps')
    
    # Model architecture options
    parser.add_argument('--amp',           action='store_true', default=False,                    help='Mixed Precision')
    parser.add_argument('--bilinear',      action='store_true', default=False,                    help='Bilinear upsampling')
    parser.add_argument('--classes',       '-c',  type=int,     default=2,                        help='Number of classes')
    parser.add_argument('--a4unet',        action='store_true', default=False,  dest='a4',       help='Enable A4Unet Architecture')
    
    # Dataset configuration
    parser.add_argument('--datasets',      '-d', type=str,      default='Brats', dest='datasets', help='Choose Dataset')
    parser.add_argument('--input_size',    '-i',  type=int,     default=128,   dest='input_size', help='Input Size of A4Unet')

    # New/Adjusted CLI parameters
    parser.add_argument('--data-root',           type=str,      default='/root/autodl-tmp/small', dest='data_root', help='Dataset root directory')
    parser.add_argument('--outdir',              type=str,      default='runs',                    help='Output root directory containing checkpoints and tensorboard logs')
    parser.add_argument('--num-workers',         type=int,      default=4,                         help='DataLoader num_workers')
    parser.add_argument('--save-best-only',      action='store_true', default=False,               help='Only save best.pth on improvement')
    parser.add_argument('--seed',                type=int,      default=0,                         help='Random seed')
    parser.add_argument('--grad-accum-steps',    type=int,      default=1,                         help='Gradient accumulation steps')
    parser.add_argument('--early-stop-patience', type=int,      default=0,                         help='Early stopping patience (0 to disable)')
    parser.add_argument('--scheduler',           type=str,      default='none', choices=['none','plateau','cosine'], help='LR scheduler type')
    parser.add_argument('--save-every',          type=int,      default=0,                         help='Save last_epochXX.pth every N epochs (0 to disable)')

    return parser.parse_args()


if __name__ == '__main__':
    # =============================================================================
    # MAIN EXECUTION BLOCK
    # =============================================================================
    
    # Parse command line arguments
    args = get_args()
    
    # Setup logging and device detection
    # loguru 默认输出到 stderr，如需文件请查看训练阶段写入的 run_dir/train.log
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device {device}!')

    # Validate val ratio
    if not (0.0 < args.val < 1.0):
        raise ValueError('--val-ratio must be in (0, 1).')
    
    # Determine input channels based on dataset type
    if args.datasets == 'Brats' or args.datasets == 'Hippo':
        input_channel = 4  # Medical images: T1, T1ce, T2, FLAIR
    else:
        input_channel = 3  # Natural images: RGB
    
    # =============================================================================
    # MODEL INITIALIZATION
    # =============================================================================
    if not args.a4:  # Standard UNet
        log.info('Model U-Net is initiating!')
        model = UNet(n_channels=input_channel, n_classes=args.classes, bilinear=args.bilinear)
        # Optimize tensor storage for better performance
        model = model.to(memory_format=torch.channels_last)
    else:  # A4-UNet architecture
        log.info('Model A4-Unet is initiating!')
        model = create_a4unet_model(
            image_size=args.input_size, 
            num_channels=128, 
            num_res_blocks=2, 
            num_classes=args.classes, 
            learn_sigma=True, 
            in_ch=input_channel
        )
    
    log.info(f'Model loaded!')
    
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
            log.info(f'Model loaded from {args.load}')
        except Exception as e:
            log.error(f'Failed to load pretrained model: {e}')
    
    # Move model to training device
    model.to(device=device)
    
    # =============================================================================
    # TRAINING EXECUTION
    # =============================================================================
    try:
        # Prepare run directories with timestamp
        run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
        outdir = Path(args.outdir)
        run_dir = outdir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        # attach file logger as early as possible
        try:
            _base_logger.add(str(run_dir / 'train.log'), enqueue=True, backtrace=False, diagnose=False)
            log.info(f'Run directory: {run_dir}')
            log.info(f'Log file: {run_dir / "train.log"}')
        except Exception:
            pass

        # Start training with all configured parameters
        train_model(
            model=model, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.lr, 
            device=device, 
            img_scale=args.scale, 
            val_percent=args.val, 
            val_step=args.valstep, 
            amp=args.amp, 
            a4unet=args.a4, 
            datasets=args.datasets, 
            input_size=args.input_size,
            data_root=Path(args.data_root),
            outdir=outdir,
            run_name=run_name,
            num_workers=args.num_workers,
            save_best_only=args.save_best_only,
            seed=args.seed,
            grad_accum_steps=args.grad_accum_steps,
            early_stop_patience=args.early_stop_patience,
            scheduler_type=args.scheduler,
            resume=args.resume,
            save_every=args.save_every
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
            val_percent=args.val, 
            val_step=args.valstep, 
            amp=args.amp, 
            a4unet=args.a4, 
            datasets=args.datasets, 
            input_size=args.input_size,
            data_root=Path(args.data_root),
            outdir=outdir,
            run_name=run_name+'-retry',
            num_workers=args.num_workers,
            save_best_only=args.save_best_only,
            seed=args.seed,
            grad_accum_steps=args.grad_accum_steps,
            early_stop_patience=args.early_stop_patience,
            scheduler_type=args.scheduler,
            resume=args.resume,
            save_every=args.save_every
        )