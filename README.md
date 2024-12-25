

# A4-Unet: A Brain Tumor Segmentation Architecture with Multi-Attention Mechanism
A4-Unet is a brain tumor segmentation architecture for Medical Image Segmentation. The algorithm is elaborated on our paper [A4-Unet: Deformable Multi-Scale Attention Network for Brain Tumor Segmentation](https://arxiv.org/pdf/2412.06088).


## A Quick Overview 

|<img align="center" width="480" height="170" src="https://github.com/WendyWAAAAANG/A4-Unet/blob/dc5b67975d3e44653a23b019c2e71b9af61a1e6d/a4unet.png">|
| **A4-Unet** |

## News

- 24-08-20. Paper [A4-Unet: Deformable Multi-Scale Attention Network for Brain Tumor Segmentation](https://arxiv.org/pdf/2412.06088) has been officially accepted by IEEE BIBM 2024 🥳


## Requirement

``pip install -r requirement.txt``

## Example Cases

### Brain Tumor Segmentation from MRI
1. Download BRATS2020 dataset from https://www.med.upenn.edu/cbica/brats2020/data.html. Your dataset folder should be like:
~~~
data
└───training
│   └───slice0001
│       │   brats_train_001_t1_123_w.nii.gz
│       │   brats_train_001_t2_123_w.nii.gz
│       │   brats_train_001_flair_123_w.nii.gz
│       │   brats_train_001_t1ce_123_w.nii.gz
│       │   brats_train_001_seg_123_w.nii.gz
│   └───slice0002
│       │  ...
└───testing
│   └───slice1000
│       │  ...
│   └───slice1001
│       │  ...
~~~
    
