import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils


class HIPPODataset3D(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for 3D Hippocampus MRI data.
    
    This dataset loads NIfTI files containing 3D MRI volumes and their corresponding
    segmentation labels, then extracts 2D slices for training/testing.
    
    Args:
        labels_tr_dir (str): Path to directory containing label NIfTI files
        image_tr_dir (str): Path to directory containing image NIfTI files  
        transform: Torchvision transforms to apply to the data
        test_flag (bool): If True, returns only images. If False, returns images and labels
    """
    
    def __init__(self, labels_tr_dir, image_tr_dir, transform=None, test_flag=False):
        super().__init__()
        
        # Expand user paths (handles ~ in paths)
        self.labels_tr_dir = os.path.expanduser(labels_tr_dir)
        self.image_tr_dir = os.path.expanduser(image_tr_dir)
        self.transform = transform
        self.test_flag = test_flag
        
        # Initialize lists to store file paths
        self.image_paths = []
        self.label_paths = []
        
        # Collect all image file paths
        self._collect_image_paths()
        
        # Collect all label file paths
        self._collect_label_paths()
        
        # Validate that we have matching numbers of images and labels
        assert len(self.image_paths) == len(self.label_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"
        
        print(f"Dataset initialized with {len(self.image_paths)} volumes")

    def _collect_image_paths(self):
        """Collect all hippocampus image file paths from the image directory."""
        for root, dirs, files in os.walk(self.image_tr_dir):
            if not dirs:  # Only process leaf directories
                files.sort()  # Ensure consistent ordering
                for filename in files:
                    if filename.startswith("hippocampus") and filename.endswith(".nii.gz"):
                        self.image_paths.append(os.path.join(root, filename))

    def _collect_label_paths(self):
        """Collect all hippocampus label file paths from the labels directory."""
        for root, dirs, files in os.walk(self.labels_tr_dir):
            if not dirs:  # Only process leaf directories
                files.sort()  # Ensure consistent ordering
                for filename in files:
                    if filename.startswith("hippocampus") and filename.endswith(".nii.gz"):
                        self.label_paths.append(os.path.join(root, filename))

    def __len__(self):
        """
        Return total number of 2D slices across all 3D volumes.
        Assumes each MRI volume has 155 slices.
        """
        return len(self.image_paths) * 155

    def __getitem__(self, index):
        """
        Get a single 2D slice and its corresponding label.
        
        Args:
            index (int): Index of the slice to retrieve
            
        Returns:
            If test_flag=True: (image_slice, filename)
            If test_flag=False: (image_slice, label_slice, filename)
        """
        # Calculate which volume and which slice within that volume
        volume_index = index // 155
        slice_index = index % 155
        
        # Load the image volume
        image_path = self.image_paths[volume_index]
        nib_img = nibabel.load(image_path)
        image_data = torch.tensor(nib_img.get_fdata())
        image_slice = image_data[:, :, slice_index]
        
        # Generate filename for this slice
        base_filename = os.path.splitext(os.path.splitext(os.path.basename(image_path))[0])[0]
        slice_filename = f"{base_filename}_slice{slice_index}.nii"
        
        if self.test_flag:
            # Test mode: return only image slice
            if self.transform:
                image_slice = self.transform(image_slice)
            return image_slice, slice_filename
        
        else:
            # Training mode: return both image and label slices
            label_path = self.label_paths[volume_index]
            nib_label = nibabel.load(label_path)
            label_data = torch.tensor(nib_label.get_fdata())
            label_slice = label_data[:, :, slice_index]
            
            # Convert multi-class labels to binary (background=0, any tumor=1)
            label_slice = torch.where(label_slice > 0, 1, 0).float()
            
            # Apply transforms if provided
            if self.transform:
                # Use same random state for both image and label to ensure consistent augmentation
                state = torch.get_rng_state()
                image_slice = self.transform(image_slice)
                torch.set_rng_state(state)
                label_slice = self.transform(label_slice)
            
            return image_slice, label_slice, slice_filename

    def get_volume_info(self, volume_index):
        """
        Get information about a specific volume.
        
        Args:
            volume_index (int): Index of the volume
            
        Returns:
            dict: Dictionary containing volume information
        """
        if volume_index >= len(self.image_paths):
            raise IndexError(f"Volume index {volume_index} out of range")
            
        return {
            'image_path': self.image_paths[volume_index],
            'label_path': self.label_paths[volume_index] if not self.test_flag else None,
            'volume_index': volume_index
        }