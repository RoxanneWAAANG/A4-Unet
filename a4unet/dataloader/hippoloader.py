import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils


class HIPPODataset3D(torch.utils.data.Dataset):
    def __init__(self, labels_tr_dir, image_tr_dir, transform, test_flag=False):
        super().__init__()
        self.labels_tr_dir = os.path.expanduser(labels_tr_dir)
        self.image_tr_dir = os.path.expanduser(image_tr_dir)
        self.transform = transform
        self.test_flag = test_flag
        self.image_paths = []  # Store image file paths
        self.label_paths = []  # Store label file paths

        # Collect image paths
        for root, dirs, files in os.walk(self.image_tr_dir):
            if not dirs:
                files.sort()
                for f in files:
                    if f.startswith("hippocampus") and f.endswith(".nii.gz"):
                        self.image_paths.append(os.path.join(root, f))

        # Collect label paths
        for root, dirs, files in os.walk(self.labels_tr_dir):
            if not dirs:
                files.sort()
                for f in files:
                    if f.startswith("hippocampus") and f.endswith(".nii.gz"):
                        self.label_paths.append(os.path.join(root, f))

        # Ensure matching number of images and labels
        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels don't match!"

    def __len__(self):
        return len(self.image_paths) * 155  # Assuming 155 slices per MRI volume

    def __getitem__(self, x):
        volume_index = x // 155
        slice_index = x % 155

        # filedict = self.label_paths[volume_index]
        label_path = self.label_paths[volume_index]  # Updated this line
        nib_img_label = nibabel.load(label_path)
        label_data = torch.tensor(nib_img_label.get_fdata())
        slice_label = label_data[:, :, slice_index]

        # if self.transform:
        #     slice_label = self.transform(slice_label)
        # if self.test_flag:
        #     return slice_label, label_path.split('.nii')[0] + "_slice" + str(slice_index) + ".nii"
        # else:
        #     label = torch.where(slice_label > 0, 1, 0).float()
        #     label = label[..., 8:-8, 8:-8]  # Crop to (224, 224)
        #     if self.transform:
        #         label = self.transform(label)
        #     return label, label_path.split('.nii')[0] + "_slice" + str(slice_index) + ".nii"


        # for i in filedict:
        #     nib_img = nibabel.load(i)
        #     path = i
        #     o = torch.tensor(nib_img.get_fdata())[:,:,slice]
        #     out.append(o)
        # out = torch.stack(out)
        
        out = slice_label
        if self.test_flag:
            image = out
            # image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return slice_image, image_path.split('.nii')[0] + "_slice" + str(slice_index) + ".nii"
        else:
            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            # label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return slice_image, label, image_path.split('.nii')[0] + "_slice" + str(slice_index) + ".nii"




            