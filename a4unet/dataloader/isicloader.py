import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class ISICDataset(Dataset):
    """
    PyTorch Dataset class for ISIC (International Skin Imaging Collaboration) dataset.
    
    This dataset loads skin lesion images and their corresponding segmentation masks
    for dermatological image analysis tasks.
    
    Args:
        img_path (str): Path to directory containing input images
        mask_path (str): Path to directory containing segmentation masks
        transform: Torchvision transforms to apply to both images and masks
        img_extension (str): File extension for input images (default: '.jpg')
        mask_suffix (str): Suffix added to mask filenames (default: '_segmentation')
        mask_extension (str): File extension for mask files (default: '.png')
    """
    
    def __init__(self, img_path, mask_path, transform=None, 
                 img_extension='.jpg', mask_suffix='_segmentation', mask_extension='.png'):
        
        # Store paths and configuration
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.img_extension = img_extension
        self.mask_suffix = mask_suffix
        self.mask_extension = mask_extension
        self.num_classes = 2  # Binary segmentation: background and lesion
        
        # Validate input directories
        self._validate_directories()
        
        # Build annotation lines (pairs of image and mask filenames)
        self.annotation_lines = self._build_annotation_lines()
        self.length = len(self.annotation_lines)
        
        print(f"Dataset initialized with {self.length} image-mask pairs")
    
    def _validate_directories(self):
        """Validate that the input directories exist."""
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f"Image directory not found: {self.img_path}")
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_path}")
    
    def _build_annotation_lines(self):
        """
        Build annotation lines that pair image filenames with mask filenames.
        
        Returns:
            list: List of strings in format "image_name mask_name"
        """
        # Get sorted lists of files in both directories
        img_files = sorted([f for f in os.listdir(self.img_path) 
                           if f.endswith(self.img_extension)])
        mask_files = sorted([f for f in os.listdir(self.mask_path) 
                            if f.endswith(self.mask_extension)])
        
        annotation_lines = []
        unmatched_images = []
        
        for img_file in img_files:
            # Extract base name without extension
            img_base_name = os.path.splitext(img_file)[0]
            
            # Construct expected mask filename
            expected_mask_file = f"{img_base_name}{self.mask_suffix}{self.mask_extension}"
            
            if expected_mask_file in mask_files:
                # Create annotation line with base names (no extensions)
                mask_base_name = os.path.splitext(expected_mask_file)[0]
                annotation_line = f"{img_base_name} {mask_base_name}"
                annotation_lines.append(annotation_line)
            else:
                unmatched_images.append(img_file)
        
        # Warn about unmatched files
        if unmatched_images:
            print(f"Warning: {len(unmatched_images)} images have no matching masks:")
            for img in unmatched_images[:5]:  # Show first 5
                print(f"  - {img}")
            if len(unmatched_images) > 5:
                print(f"  ... and {len(unmatched_images) - 5} more")
        
        if not annotation_lines:
            raise ValueError("No matching image-mask pairs found!")
        
        return annotation_lines
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.length
    
    def __getitem__(self, index):
        """
        Get a single image-mask pair.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, mask, filename) where:
                - image: PIL Image or transformed tensor
                - mask: PIL Image or transformed tensor  
                - filename: Base filename without extension
        """
        if index >= self.length:
            raise IndexError(f"Index {index} out of range for dataset of size {self.length}")
        
        # Parse annotation line to get filenames
        annotation_line = self.annotation_lines[index]
        img_name, mask_name = annotation_line.split()
        
        # Construct full file paths
        img_path = os.path.join(self.img_path, f"{img_name}{self.img_extension}")
        mask_path = os.path.join(self.mask_path, f"{mask_name}{self.mask_extension}")
        
        # Load image and mask
        try:
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')  # Grayscale for segmentation mask
        except Exception as e:
            raise IOError(f"Error loading files for index {index}: {e}")
        
        # Apply transforms with synchronized random state
        if self.transform:
            # Save random state to ensure same augmentation for image and mask
            random_state = torch.get_rng_state()
            img = self.transform(img)
            
            # Restore random state for mask to get same augmentation
            torch.set_rng_state(random_state)
            mask = self.transform(mask)
        
        return img, mask, img_name
    
    def get_sample_info(self, index):
        """
        Get detailed information about a specific sample.
        
        Args:
            index (int): Index of the sample
            
        Returns:
            dict: Dictionary containing sample information
        """
        if index >= self.length:
            raise IndexError(f"Index {index} out of range")
        
        annotation_line = self.annotation_lines[index]
        img_name, mask_name = annotation_line.split()
        
        img_path = os.path.join(self.img_path, f"{img_name}{self.img_extension}")
        mask_path = os.path.join(self.mask_path, f"{mask_name}{self.mask_extension}")
        
        return {
            'index': index,
            'img_name': img_name,
            'mask_name': mask_name,
            'img_path': img_path,
            'mask_path': mask_path,
            'img_exists': os.path.exists(img_path),
            'mask_exists': os.path.exists(mask_path)
        }
    
    def verify_dataset_integrity(self):
        """
        Verify that all image-mask pairs exist and can be loaded.
        
        Returns:
            dict: Summary of dataset integrity check
        """
        missing_images = []
        missing_masks = []
        corrupted_files = []
        
        print("Verifying dataset integrity...")
        
        for i in range(len(self)):
            info = self.get_sample_info(i)
            
            if not info['img_exists']:
                missing_images.append(info['img_path'])
            if not info['mask_exists']:
                missing_masks.append(info['mask_path'])
            
            # Try to load files
            try:
                if info['img_exists'] and info['mask_exists']:
                    img = Image.open(info['img_path'])
                    mask = Image.open(info['mask_path'])
                    img.verify()
                    mask.verify()
            except Exception as e:
                corrupted_files.append((info['img_path'], str(e)))
        
        summary = {
            'total_samples': len(self),
            'missing_images': len(missing_images),
            'missing_masks': len(missing_masks),
            'corrupted_files': len(corrupted_files),
            'is_valid': len(missing_images) == 0 and len(missing_masks) == 0 and len(corrupted_files) == 0
        }
        
        print(f"Dataset integrity check complete:")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Missing images: {summary['missing_images']}")
        print(f"  Missing masks: {summary['missing_masks']}")
        print(f"  Corrupted files: {summary['corrupted_files']}")
        print(f"  Dataset valid: {summary['is_valid']}")
        
        return summary