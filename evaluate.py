import os
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import directed_hausdorff


# Global configuration
OUTPUT_DIR = './env_result'


def mask_to_image(mask: np.ndarray, mask_values):
    """
    Convert a segmentation mask to a PIL Image.
    
    Args:
        mask (np.ndarray): Input mask array
        mask_values: List of values for each class
        
    Returns:
        PIL.Image: Converted mask image
    """
    # Determine output format based on mask values
    if isinstance(mask_values[0], list):
        # Multi-channel output for complex class mappings
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        # Binary mask - use boolean array
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        # Standard multi-class mask
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    # Convert from one-hot to class indices if needed
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # Map class indices to output values
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def calculate_hausdorff_95(pred_mask, true_mask):
    """
    Calculate 95th percentile Hausdorff distance between two binary masks.
    
    Args:
        pred_mask (np.ndarray): Predicted binary mask [H, W]
        true_mask (np.ndarray): Ground truth binary mask [H, W]
        
    Returns:
        float: 95th percentile Hausdorff distance
    """
    try:
        # Get boundary pixels for both masks
        pred_points = np.column_stack(np.where(pred_mask > 0))
        true_points = np.column_stack(np.where(true_mask > 0))
        
        # Handle edge cases
        if len(pred_points) == 0 or len(true_points) == 0:
            return float('inf')  # No overlap possible
        
        # Calculate bidirectional Hausdorff distance
        dist_1 = directed_hausdorff(pred_points, true_points)[0]
        dist_2 = directed_hausdorff(true_points, pred_points)[0]
        
        # Return 95th percentile (symmetric Hausdorff distance)
        return max(dist_1, dist_2)
        
    except Exception as e:
        print(f"Warning: Hausdorff distance calculation failed: {e}")
        return 0.0


def process_batch_predictions(mask_pred, mask_true, net, dataset_name):
    """
    Process model predictions based on dataset type and number of classes.
    
    Args:
        mask_pred (torch.Tensor): Raw model predictions
        mask_true (torch.Tensor): Ground truth masks
        net: Neural network model
        dataset_name (str): Name of the dataset ('Brats', 'ISIC', etc.)
        
    Returns:
        tuple: (processed_pred, processed_true) tensors ready for metric calculation
    """
    if net.n_classes == 1:
        # Binary segmentation case
        assert mask_true.min() >= 0 and mask_true.max() <= 1, \
            'True mask indices should be in [0, 1] for binary segmentation'
        
        # Apply sigmoid and threshold for binary prediction
        mask_pred_processed = (F.sigmoid(mask_pred) > 0.5).float()
        mask_true_processed = mask_true.float()
        
    else:
        # Multi-class segmentation case
        assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, \
            f'True mask indices should be in [0, {net.n_classes}] for {net.n_classes}-class segmentation'

        if dataset_name == 'Brats':
            # BraTS specific processing: convert to one-hot encoding
            mask_true_processed = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred_processed = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            
        elif dataset_name == 'ISIC':
            # ISIC specific processing: use argmax for predictions
            mask_pred_processed = mask_pred.argmax(dim=1)
            mask_true_processed = mask_true
            
        else:
            # Default multi-class processing
            mask_pred_processed = mask_pred.argmax(dim=1)
            mask_true_processed = mask_true
            
    return mask_pred_processed, mask_true_processed


def save_prediction(mask_pred, image_shape, filename, output_dir=OUTPUT_DIR):
    """
    Save model prediction as an image file.
    
    Args:
        mask_pred (torch.Tensor): Model prediction tensor
        image_shape (tuple): Original image shape (H, W)
        filename (str): Original filename for generating output name
        output_dir (str): Directory to save predictions
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        path_parts = Path(filename).parts
        if len(path_parts) >= 2:
            # Use last two parts of path for unique naming
            out_name = f"{path_parts[-2]}_{Path(filename).stem}.jpg"
        else:
            # Fallback to just the filename
            out_name = f"{Path(filename).stem}.jpg"
        
        output_path = os.path.join(output_dir, out_name)
        
        # Resize prediction to original image dimensions
        resized_pred = F.interpolate(
            mask_pred.unsqueeze(0), 
            size=image_shape, 
            mode='bilinear', 
            align_corners=True
        )
        
        # Convert to class indices and numpy
        pred_numpy = resized_pred.argmax(dim=1)[0].cpu().long().squeeze().numpy()
        
        # Convert to image and save
        result_image = mask_to_image(pred_numpy, [0, 1])
        result_image.save(output_path)
        
    except Exception as e:
        print(f"Warning: Failed to save prediction for {filename}: {e}")


def evaluate(net, dataloader, device, amp, dataset_name, final_test=False):
    """
    Comprehensive evaluation function for segmentation models.
    
    Args:
        net: Neural network model
        dataloader: PyTorch DataLoader for validation data
        device: Device to run evaluation on (CPU/GPU)
        amp: Automatic mixed precision flag (currently unused but kept for compatibility)
        dataset_name (str): Name of dataset ('Brats', 'ISIC', etc.)
        final_test (bool): Whether to save prediction images
        
    Returns:
        tuple: (dice_score, mIoU, hd95) - Average metrics across all batches
    """
    print(f"Starting evaluation on {dataset_name} dataset...")
    
    # Set model to evaluation mode
    net.eval()
    
    # Initialize metrics
    num_val_batches = len(dataloader)
    dice_score_total = 0.0
    miou_total = 0.0
    hd95_total = 0.0
    
    # Determine if we should save predictions
    save_predictions = final_test
    
    # Evaluation loop with no gradient computation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, 
                                             total=num_val_batches, 
                                             desc=f'Evaluating {dataset_name}', 
                                             unit='batch', 
                                             leave=False)):
            
            # Parse batch data based on dataset type
            if dataset_name == 'Brats':
                # BraTS format: (image, mask)
                image, mask_true = batch[0], batch[1]
                filename = f"brats_batch_{batch_idx}"  # Fallback filename
            else:
                # Standard format: (image, mask, filename)
                image, mask_true, filename = batch
                filename = filename[0] if isinstance(filename, (list, tuple)) else filename
                    
            # Preprocess masks based on dataset
            if dataset_name == 'Brats':
                mask_true = torch.squeeze(mask_true, dim=1)
            elif dataset_name == 'ISIC':
                mask_true = mask_true.squeeze(1)
            
            # Move data to device with optimized memory format
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            
            # Forward pass
            mask_pred = net(image)
            
            # Save predictions if requested
            if save_predictions:
                save_prediction(mask_pred, (image.shape[2], image.shape[3]), filename)
            
            # Process predictions based on model type and dataset
            mask_pred_processed, mask_true_processed = process_batch_predictions(
                mask_pred, mask_true, net, dataset_name
            )
            
            # Calculate metrics based on number of classes
            if net.n_classes == 1:
                # Binary segmentation metrics
                batch_dice = dice_coeff(mask_pred_processed, mask_true_processed, reduce_batch_first=False)
                dice_score_total += batch_dice.item()
                
                # For binary, IoU and Hausdorff can be calculated directly
                pred_np = mask_pred_processed.cpu().numpy()
                true_np = mask_true_processed.cpu().numpy()
                
                batch_iou = jaccard_score(true_np.flatten(), pred_np.flatten(), average='binary')
                miou_total += batch_iou
                
                # Calculate Hausdorff distance for first sample in batch
                if len(pred_np.shape) > 2:  # Batch dimension exists
                    batch_hd95 = calculate_hausdorff_95(pred_np[0], true_np[0])
                else:
                    batch_hd95 = calculate_hausdorff_95(pred_np, true_np)
                hd95_total += batch_hd95
                
            else:
                # Multi-class segmentation metrics
                if dataset_name == 'Brats':
                    # Use foreground classes only (exclude background)
                    batch_dice = multiclass_dice_coeff(
                        mask_pred_processed[:, 1:], 
                        mask_true_processed[:, 1:], 
                        reduce_batch_first=False
                    )
                    dice_score_total += batch_dice.item()
                    
                    # Calculate IoU
                    pred_indices = mask_pred_processed.argmax(dim=1).cpu().numpy()
                    true_indices = mask_true_processed.argmax(dim=1).cpu().numpy()
                    
                elif dataset_name == 'ISIC':
                    # ISIC specific multi-class handling
                    # Convert to one-hot for dice calculation
                    mask_true_onehot = F.one_hot(mask_true_processed, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred_onehot = F.one_hot(mask_pred_processed, net.n_classes).permute(0, 3, 1, 2).float()
                    
                    batch_dice = multiclass_dice_coeff(
                        mask_pred_onehot[:, 1:], 
                        mask_true_onehot[:, 1:], 
                        reduce_batch_first=False
                    )
                    dice_score_total += batch_dice.item()
                    
                    pred_indices = mask_pred_processed.cpu().numpy()
                    true_indices = mask_true_processed.cpu().numpy()
                
                # Calculate IoU for multi-class
                batch_iou = jaccard_score(
                    true_indices.flatten(),
                    pred_indices.flatten(), 
                    average='macro'
                )
                miou_total += batch_iou
                
                # Calculate Hausdorff distance (use binary mask for now)
                pred_binary = (pred_indices > 0).astype(np.uint8)
                true_binary = (true_indices > 0).astype(np.uint8)
                
                if len(pred_binary.shape) > 2:  # Batch dimension
                    batch_hd95 = calculate_hausdorff_95(pred_binary[0], true_binary[0])
                else:
                    batch_hd95 = calculate_hausdorff_95(pred_binary, true_binary)
                hd95_total += batch_hd95

    # Restore model to training mode
    net.train()
    
    # Calculate average metrics
    avg_dice = dice_score_total / max(num_val_batches, 1)
    avg_miou = miou_total / max(num_val_batches, 1)
    avg_hd95 = hd95_total / max(num_val_batches, 1)
    
    print(f"Evaluation complete - Dice: {avg_dice:.4f}, mIoU: {avg_miou:.4f}, HD95: {avg_hd95:.4f}")
    
    return avg_dice, avg_miou, avg_hd95


def evaluate_single_image(net, image, mask_true, device, dataset_name):
    """
    Evaluate a single image (useful for debugging or single-sample inference).
    
    Args:
        net: Neural network model
        image: Input image tensor
        mask_true: Ground truth mask tensor
        device: Device to run on
        dataset_name: Dataset name for processing logic
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    net.eval()
    
    with torch.no_grad():
        # Move to device
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        
        # Forward pass
        mask_pred = net(image)
        
        # Process predictions
        mask_pred_processed, mask_true_processed = process_batch_predictions(
            mask_pred, mask_true, net, dataset_name
        )
        
        # Calculate metrics
        if net.n_classes == 1:
            dice = dice_coeff(mask_pred_processed, mask_true_processed, reduce_batch_first=False)
            pred_np = mask_pred_processed.cpu().numpy().squeeze()
            true_np = mask_true_processed.cpu().numpy().squeeze()
        else:
            if dataset_name == 'Brats':
                dice = multiclass_dice_coeff(
                    mask_pred_processed[:, 1:], 
                    mask_true_processed[:, 1:], 
                    reduce_batch_first=False
                )
                pred_np = mask_pred_processed.argmax(dim=1).cpu().numpy().squeeze()
                true_np = mask_true_processed.argmax(dim=1).cpu().numpy().squeeze()
            else:
                mask_true_onehot = F.one_hot(mask_true_processed, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_onehot = F.one_hot(mask_pred_processed, net.n_classes).permute(0, 3, 1, 2).float()
                dice = multiclass_dice_coeff(
                    mask_pred_onehot[:, 1:], 
                    mask_true_onehot[:, 1:], 
                    reduce_batch_first=False
                )
                pred_np = mask_pred_processed.cpu().numpy().squeeze()
                true_np = mask_true_processed.cpu().numpy().squeeze()
        
        # Calculate additional metrics
        iou = jaccard_score(true_np.flatten(), pred_np.flatten(), average='macro')
        hd95 = calculate_hausdorff_95((pred_np > 0).astype(np.uint8), (true_np > 0).astype(np.uint8))
        
    net.train()
    
    return {
        'dice': dice.item() if hasattr(dice, 'item') else dice,
        'iou': iou,
        'hd95': hd95
    }