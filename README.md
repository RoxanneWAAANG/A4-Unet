

# A4-Unet: A Brain Tumor Segmentation Architecture with Multi-Attention Mechanism
A4-Unet is a brain tumor segmentation architecture for Medical Image Segmentation. The algorithm is elaborated on our paper [A4-Unet: Deformable Multi-Scale Attention Network for Brain Tumor Segmentation](https://arxiv.org/pdf/2412.06088).


## A Quick Overview 

<img align="center" width="370" height="210" src="https://github.com/WendyWAAAAANG/A4-Unet/blob/dc5b67975d3e44653a23b019c2e71b9af61a1e6d/a4unet.png">

**A4-Unet** 

***Architecture***
The overall framework of A4-Unet is structured as follows:
* Encoder: Large-kernel convolutions with variable sizes to capture multi-scale information.
* Bottleneck: SSPP for hierarchical feature extraction.
* Decoder: DCT-based attention combined with skip connections for refined segmentation output.

***Contributions & Features***
* Multi-scale feature extraction using large-kernel variable convolutions.
* Swin Spatial Pyramid Pooling (SSPP) for global and local feature integration.
* DCT-based channel attention for enhanced decoder performance.
* Efficient segmentation with skip connections and attention-based aggregation.

## News

- 24-08-20. Paper [A4-Unet: Deformable Multi-Scale Attention Network for Brain Tumor Segmentation](https://arxiv.org/pdf/2412.06088) has been officially accepted by IEEE BIBM 2024 🥳


## Installation
1. Ensure you have Python 3.8+ and PyTorch installed.
2. ``git clone https://github.com/WendyWAAAAANG/A4-Unet.git``
3. ``cd A4-Unet``
4. ``pip install -r requirements.txt``

## Configuration

## Model Checkpoint

## Citation
If you use A4-Unet in your research, please cite the following:

@article{wang2024a4,
  title={A4-Unet: Deformable Multi-Scale Attention Network for Brain Tumor Segmentation},
  author={Wang, Ruoxin and Tang, Tianyi and Du, Haiming and Cheng, Yuxuan and Wang, Yu and Yang, Lingjie and Duan, Xiaohui and Yu, Yunfang and Zhou, Yu and Chen, Donglong},
  journal={arXiv preprint arXiv:2412.06088},
  year={2024}
}

## License
This project is licensed under the MIT License - see the License.md file for details.

## Ethical Statement
This repository accompanies the research paper "A4-Unet: A Brain Tumor Segmentation Architecture with Multi-Attention Mechanism" published in IEEE. The following ethical considerations were observed:

1. **Data Usage**: The datasets used in this research are publicly available and comply with their respective licensing agreements. No personal or sensitive data was used.

2. **Reproducibility**: All code and experimental settings are shared in this repository to support reproducibility of results. Instructions for running the experiments are included.

3. **Conflict of Interest**: The authors declare no conflicts of interest in the research and development of this project.

4. **Social Responsibility**: The findings of this study are intended for advancing the field of computer vision and are not designed or recommended for harmful applications.


## Contributing
We welcome contributions! Please read the Contribute.md file for guidelines on submitting issues and pull requests.

## Contact
For questions, feel free to email or create an issue in this repository.

**Author:** Ruoxin Wang, Tianyi Tang, Haiming Du

**Email:** ruoxinwaaang@gmail.com, trumantytang@163.com, jennyduuu@163.com

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


    
** Few Sections of the README are re-articulated using ChatGPT.
