import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils


class BRATSDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for BraTS (Brain Tumor Segmentation) dataset - Full 3D volumes.
    
    This dataset loads multi-modal brain MRI scans and their corresponding tumor segmentations.
    Each datapoint consists of 4 MRI modalities (T1, T1ce, T2, FLAIR) and segmentation masks.
    
    Expected directory structure:
        - Subfolders containing files named like: brats_train_001_XXX_123_w.nii.gz
        - Where XXX is one of: t1, t1ce, t2, flair, seg
        - All five files with the same base name belong to the same patient
        
    Args:
        directory (str): Root directory containing BraTS data
        transform: Torchvision transforms to apply to the data
        test_flag (bool): If True, only loads imaging data (no segmentation)
    """
    
    def __init__(self, directory, transform=None, test_flag=False):
        super().__init__()
        
        # Expand user path and store configuration
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.test_flag = test_flag

        # Define MRI sequence types based on test/train mode
        if test_flag:
            # Test mode: only imaging sequences (no segmentation)
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            # Train mode: imaging sequences + segmentation
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        
        # Build database of file paths
        self.database = self._build_database()
        
        print(f"BRATSDataset initialized with {len(self.database)} volumes")

    def _build_database(self):
        """
        Scan directory structure and build database of file paths.
        
        Returns:
            list: List of dictionaries, each containing paths for all sequences of one patient
        """
        database = []
        
        for root, dirs, files in os.walk(self.directory):
            # Process only leaf directories (containing actual files)
            if not dirs:
                files.sort()  # Ensure consistent ordering
                datapoint = {}
                
                # Extract sequence type from each filename and store path
                for filename in files:
                    if filename.endswith('.nii.gz'):
                        # Extract sequence type from filename (4th component after splitting by '_')
                        try:
                            seqtype = filename.split('_')[3]
                            datapoint[seqtype] = os.path.join(root, filename)
                        except IndexError:
                            print(f"Warning: Unexpected filename format: {filename}")
                            continue
                
                # Validate that all required sequences are present
                if set(datapoint.keys()) == self.seqtypes_set:
                    database.append(datapoint)
                else:
                    missing = self.seqtypes_set - set(datapoint.keys())
                    extra = set(datapoint.keys()) - self.seqtypes_set
                    print(f"Warning: Incomplete datapoint in {root}")
                    if missing:
                        print(f"  Missing: {missing}")
                    if extra:
                        print(f"  Extra: {extra}")
        
        if not database:
            raise ValueError(f"No valid datapoints found in {self.directory}")
            
        return database

    def __len__(self):
        """Return the total number of 3D volumes in the dataset."""
        return len(self.database)

    def __getitem__(self, index):
        """
        Get a single 3D volume with all MRI modalities.
        
        Args:
            index (int): Index of the volume to retrieve
            
        Returns:
            tuple: (image, label, path) where:
                - image: Tensor of shape [n_modalities, H, W, D] (imaging data)
                - label: Tensor of shape [1, H, W, D] (segmentation) or same as image if test_flag
                - path: String path to one of the files (for reference)
        """
        if index >= len(self.database):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.database)}")
        
        # Load all sequences for this patient
        filedict = self.database[index]
        sequence_data = []
        reference_path = None
        
        for seqtype in self.seqtypes:
            filepath = filedict[seqtype]
            nib_img = nibabel.load(filepath)
            volume_data = torch.tensor(nib_img.get_fdata())
            sequence_data.append(volume_data)
            
            # Store path for reference (use first sequence)
            if reference_path is None:
                reference_path = filepath
        
        # Stack all sequences into single tensor
        stacked_data = torch.stack(sequence_data)
        
        if self.test_flag:
            # Test mode: return imaging data only
            image = stacked_data
            
            # Crop to standard size (224, 224) - removes 8 pixels from each border
            image = image[..., 8:-8, 8:-8]
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            # Return (image, image, path) - duplicate image for consistency
            return image, image, reference_path
            
        else:
            # Training mode: separate imaging data from segmentation
            image = stacked_data[:-1, ...]  # All sequences except last (segmentation)
            label = stacked_data[-1, ...][None, ...]  # Last sequence (segmentation) with added channel dim
            
            # Crop both image and label to standard size
            image = image[..., 8:-8, 8:-8]
            label = label[..., 8:-8, 8:-8]
            
            # Convert multi-class segmentation to binary (background=0, tumor=1)
            label = torch.where(label > 0, 1, 0).float()
            
            # Apply synchronized transforms if provided
            if self.transform:
                # Use same random state for image and label to ensure consistent augmentation
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            
            return image, label, reference_path


class BRATSDataset3D(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for BraTS dataset - 2D slice extraction from 3D volumes.
    
    This dataset extracts 2D slices from 3D brain MRI volumes for slice-based training/inference.
    Each 3D volume is assumed to have 155 slices, creating 155 * num_volumes total samples.
    
    Expected directory structure:
        - Subfolders containing files with BraTS naming convention
        - Supports both old format and BraTS 2021 format
        
    Args:
        directory (str): Root directory containing BraTS data
        transform: Torchvision transforms to apply to the data
        test_flag (bool): If True, only loads imaging data (no segmentation)
    """
    
    def __init__(self, directory, transform=None, test_flag=False):
        super().__init__()
        
        # Expand user path and store configuration
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.test_flag = test_flag
        self.mask_values = [0, 1]  # Binary segmentation values
        self.slices_per_volume = 155  # Standard number of slices in BraTS volumes

        # Define MRI sequence types based on dataset version and mode
        # if test_flag:
        #     self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        # else:
        #     self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']
        
        # if BraTS 2021 #
        if test_flag:
            # Test mode: only imaging sequences
            self.seqtypes = ['t1n', 't1c', 't2w', 't2f']
        else:
            # Train mode: imaging sequences + segmentation
            self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        # ============= #

        self.seqtypes_set = set(self.seqtypes)
        
        # Build database of file paths
        self.database = self._build_database()
        
        print(f"BRATSDataset3D initialized with {len(self.database)} volumes, "
              f"{len(self)} total slices")

    def _build_database(self):
        """
        Scan directory structure and build database of file paths.
        
        Returns:
            list: List of dictionaries, each containing paths for all sequences of one patient
        """
        database = []
        
        for root, dirs, files in os.walk(self.directory):
            # Process only leaf directories (containing actual files)
            if not dirs:
                files.sort()  # Ensure consistent ordering
                datapoint = {}
                
                # Extract sequence type from each filename
                for filename in files:
                    if filename.endswith('.nii.gz'):
                        try:
                            # seqtype = f.split('_')[3].split('.')[0]
                            # if BraTS 2021 #
                            # BraTS 2021 naming: extract sequence type from 5th component (0-indexed 4th)
                            seqtype = filename.split('-')[4].split('.')[0]
                            # ============= #
                            datapoint[seqtype] = os.path.join(root, filename)
                        except IndexError:
                            print(f"Warning: Unexpected filename format: {filename}")
                            continue
                
                # Validate that all required sequences are present
                if set(datapoint.keys()) == self.seqtypes_set:
                    database.append(datapoint)
                else:
                    missing = self.seqtypes_set - set(datapoint.keys())
                    extra = set(datapoint.keys()) - self.seqtypes_set
                    print(f"Warning: Incomplete datapoint in {root}")
                    if missing:
                        print(f"  Missing: {missing}")
                    if extra:
                        print(f"  Extra: {extra}")
        
        if not database:
            raise ValueError(f"No valid datapoints found in {self.directory}")
            
        return database

    def __len__(self):
        """
        Return total number of 2D slices across all 3D volumes.
        Assumes each volume has 155 slices.
        """
        return len(self.database) * self.slices_per_volume

    def __getitem__(self, index):
        """
        Get a single 2D slice from the dataset.
        
        Args:
            index (int): Global slice index across all volumes
            
        Returns:
            tuple: (image, label, virtual_path) where:
                - image: Tensor of shape [n_modalities, H, W] (2D slice)
                - label: Tensor of shape [1, H, W] (2D mask) or same as image if test_flag
                - virtual_path: String indicating slice location for reference
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        
        # Calculate which volume and which slice within that volume
        volume_index = index // self.slices_per_volume
        slice_index = index % self.slices_per_volume
        
        # Load all sequences for this patient
        filedict = self.database[volume_index]
        sequence_slices = []
        reference_path = None
        
        for seqtype in self.seqtypes:
            filepath = filedict[seqtype]
            nib_img = nibabel.load(filepath)
            
            # Extract specific slice from 3D volume
            volume_data = torch.tensor(nib_img.get_fdata())
            slice_data = volume_data[:, :, slice_index]
            sequence_slices.append(slice_data)
            
            # Store path for reference (use first sequence)
            if reference_path is None:
                reference_path = filepath
        
        # Stack all sequence slices into single tensor
        stacked_slices = torch.stack(sequence_slices)
        
        # Create virtual path indicating slice location
        base_filename = reference_path.split('.nii')[0]
        virtual_path = f"{base_filename}_slice{slice_index}.nii"
        
        if self.test_flag:
            # Test mode: return imaging data only
            image = stacked_slices
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            # Return (image, image, virtual_path) - duplicate image for consistency
            return image, image, virtual_path
            
        else:
            # Training mode: separate imaging data from segmentation
            image = stacked_slices[:-1, ...]  # All sequences except last
            label = stacked_slices[-1, ...][None, ...]  # Last sequence with added channel dim
            
            # Convert multi-class segmentation to binary
            label = torch.where(label > 0, 1, 0).float()
            
            # Apply synchronized transforms if provided
            if self.transform:
                # Use same random state for image and label
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            
            return image, label, virtual_path

    def get_volume_slice_info(self, index):
        """
        Get information about which volume and slice an index corresponds to.
        
        Args:
            index (int): Global slice index
            
        Returns:
            dict: Information about the volume and slice
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")
        
        volume_index = index // self.slices_per_volume
        slice_index = index % self.slices_per_volume
        
        return {
            'global_index': index,
            'volume_index': volume_index,
            'slice_index': slice_index,
            'total_volumes': len(self.database),
            'slices_per_volume': self.slices_per_volume
        }