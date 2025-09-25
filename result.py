import torch
import sys
import os
from pathlib import Path
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import DataLoader

# Import custom modules
from evaluate import evaluate
from a4unet.dataloader.bratsloader import BRATSDataset3D
from a4unet.a4unet import create_a4unet_model


def load_model_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from checkpoint file with proper error handling.
    
    Args:
        model: The neural network model
        checkpoint_path (Path): Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        model: Model with loaded weights
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # If checkpoint contains optimizer state, etc.
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            # If checkpoint is wrapped in 'state_dict' key
            state_dict = checkpoint['state_dict']
        else:
            # If checkpoint is just the state dict
            state_dict = checkpoint
        
        # Clean the state dict - remove unwanted keys
        cleaned_state_dict = OrderedDict()
        for key, value in state_dict.items():
            # Skip non-model parameters
            if key == 'mask_values':
                continue
            # Handle potential module wrapper prefixes
            clean_key = key.replace('module.', '') if key.startswith('module.') else key
            cleaned_state_dict[clean_key] = value
        
        # Load the cleaned state dict
        model.load_state_dict(cleaned_state_dict, strict=True)
        print("✓ Checkpoint loaded successfully")
        
        return model
        
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in checkpoint: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print("This might be due to architecture mismatch between saved and current model")
        sys.exit(1)


def create_data_loader(dataset_path, transform, test_flag=False, batch_size=1, num_workers=4):
    """
    Create a DataLoader for the BraTS dataset.
    
    Args:
        dataset_path (str): Path to the dataset directory
        transform: Transformations to apply to the data
        test_flag (bool): Whether this is test mode (no labels)
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    print(f"Creating dataset from: {dataset_path}")
    
    # Verify dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    try:
        # Create the dataset
        dataset = BRATSDataset3D(dataset_path, transform, test_flag=test_flag)
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Create the DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False  # Include all samples in evaluation
        )
        
        print(f"✓ DataLoader created with batch_size={batch_size}, num_workers={num_workers}")
        return dataloader
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        sys.exit(1)


def main():
    """Main evaluation function."""
    
    # =============================================================================
    # CONFIGURATION - Modify these paths according to your setup
    # =============================================================================
    
    # Dataset and model paths - MODIFY THESE ACCORDING TO YOUR SETUP
    DATASET_PATH = './data/MICCAI_BraTS2020_TrainingData'  # Path to your BraTS dataset
    VALIDATION_PATH = 'testset'  # Can be relative to DATASET_PATH or absolute
    CHECKPOINT_PATH = './checkpoints/checkpoint_epoch5.pth'  # Path to your trained model checkpoint
    OUTPUT_DIR = './evaluate_result'  # Directory to save evaluation results
    
    # Model parameters
    MODEL_CONFIG = {
        'image_size': 128,
        'num_channels': 128,
        'num_res_blocks': 2,
        'num_classes': 2,
        'learn_sigma': True,
        'in_ch': 4  # 4 MRI modalities for BraTS
    }
    
    # Evaluation parameters
    BATCH_SIZE = 1  # Usually 1 for evaluation to handle different image sizes
    NUM_WORKERS = 4  # Adjust based on your system
    DATASET_NAME = 'Brats'  # Must match what your evaluate function expects
    FINAL_TEST = True  # Set to True to save prediction images
    
    # =============================================================================
    # SETUP
    # =============================================================================
    
    print("=" * 60)
    print("BraTS Model Evaluation Script")
    print("=" * 60)
    
    # Check if paths exist
    paths_to_check = [DATASET_PATH, CHECKPOINT_PATH]
    if VALIDATION_PATH and not os.path.isabs(VALIDATION_PATH):
        validation_full_path = os.path.join(DATASET_PATH, VALIDATION_PATH)
    else:
        validation_full_path = VALIDATION_PATH
    
    paths_to_check.append(validation_full_path)
    
    for path in paths_to_check:
        if not os.path.exists(path):
            print(f"Error: Path does not exist: {path}")
            return
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # =============================================================================
    # MODEL SETUP
    # =============================================================================
    
    print("\n" + "-" * 40)
    print("Setting up model...")
    print("-" * 40)
    
    try:
        # Create the model
        model = create_a4unet_model(**MODEL_CONFIG)
        print(f"✓ Model created: {model.__class__.__name__}")
        
        # Load checkpoint
        model = load_model_checkpoint(model, Path(CHECKPOINT_PATH), device)
        
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
    except Exception as e:
        print(f"Error setting up model: {e}")
        return
    
    # =============================================================================
    # DATA SETUP
    # =============================================================================
    
    print("\n" + "-" * 40)
    print("Setting up data...")
    print("-" * 40)
    
    # Define data transformations
    # Note: For evaluation, typically minimal transforms are used
    transform_list = [
        transforms.Resize((MODEL_CONFIG['image_size'], MODEL_CONFIG['image_size']), antialias=True)
    ]
    transform = transforms.Compose(transform_list)
    print(f"Transforms: {[type(t).__name__ for t in transform_list]}")
    
    # Create DataLoader
    try:
        dataloader = create_data_loader(
            dataset_path=validation_full_path,
            transform=transform,
            test_flag=False,  # Set to True if you don't have ground truth labels
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        return
    
    # =============================================================================
    # EVALUATION
    # =============================================================================
    
    print("\n" + "-" * 40)
    print("Starting evaluation...")
    print("-" * 40)
    
    try:
        # Update the global output directory in the evaluate module
        import evaluate
        evaluate.OUTPUT_DIR = OUTPUT_DIR
        
        # Run evaluation
        dice_score, miou, hd95 = evaluate(
            net=model,
            dataloader=dataloader,
            device=device,
            amp=False,  # Set to True if you want to use automatic mixed precision
            dataset_name=DATASET_NAME,  # Fixed parameter name
            final_test=FINAL_TEST
        )
        
        # =============================================================================
        # RESULTS
        # =============================================================================
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Dataset: {DATASET_NAME}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Checkpoint: {Path(CHECKPOINT_PATH).name}")
        print(f"Total samples evaluated: {len(dataloader.dataset)}")
        print(f"Batch size: {BATCH_SIZE}")
        print("-" * 60)
        print(f"Dice Score: {dice_score:.4f}")
        print(f"Mean IoU:   {miou:.4f}")
        print(f"HD95:       {hd95:.4f}")
        print("=" * 60)
        
        if FINAL_TEST:
            print(f"\n✓ Prediction images saved to: {OUTPUT_DIR}")
        
        # Save results to file
        results_file = os.path.join(OUTPUT_DIR, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"BraTS Evaluation Results\n")
            f.write(f"========================\n")
            f.write(f"Dataset: {DATASET_NAME}\n")
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"Checkpoint: {Path(CHECKPOINT_PATH).name}\n")
            f.write(f"Total samples: {len(dataloader.dataset)}\n")
            f.write(f"Dice Score: {dice_score:.4f}\n")
            f.write(f"Mean IoU: {miou:.4f}\n")
            f.write(f"HD95: {hd95:.4f}\n")
        
        print(f"✓ Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()