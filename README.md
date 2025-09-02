# A4-Unet: Deformable Multi-Scale Attention Network for Brain Tumor Segmentation

A4-Unet is a novel brain tumor segmentation architecture that addresses MRI complexity and variability challenges, including irregular shapes and unclear boundaries. Our method achieves **94.4% Dice score** on the BraTS 2020 dataset, establishing new state-of-the-art benchmarks.

**Paper:** [A4-Unet: Deformable Multi-Scale Attention Network for Brain Tumor Segmentation](https://arxiv.org/pdf/2412.06088) (IEEE BIBM 2024)

## Architecture Overview

<img align="center" width="370" height="210" src="https://github.com/WendyWAAAAANG/A4-Unet/blob/dc5b67975d3e44653a23b019c2e71b9af61a1e6d/a4unet.png">

A4-Unet integrates **four advanced components** that address the key challenges in medical image segmentation:

| Component | Location | Key Innovation | Purpose |
|-----------|----------|----------------|---------|
| **DLKA** | Encoder | Deformable Large Kernel Attention | Multi-scale tumor capture |
| **SSPP** | Bottleneck | Swin Spatial Pyramid Pooling | Long-distance dependencies |
| **CAM** | Decoder | Combined Attention with DCT | Enhanced spatial-channel weighting |
| **AG** | Skip Connections | Attention Gates | Background suppression |

### Core Design Principles

Based on established semantic segmentation principles, A4-Unet addresses three critical requirements:

1. **Strong Encoder** (✓ DLKA + SSPP): Captures complex brain structures and tumor variability
2. **Multi-Scale Interaction** (✓ DLKA + SSPP): Handles significant size and shape disparities  
3. **Attention Mechanisms** (✓ CAM): Identifies crucial channels and spatial locations

## Technical Innovations

### 1. Deformable Large Kernel Attention (DLKA)
- **Purpose**: Dynamically modifies convolutional weight coefficients and deformation offsets during training
- **Innovation**: Large-kernel variable convolutions with low complexity
- **Advantage**: Better capture of multi-scale information and irregular tumor shapes
- **Components**: Deformable Convolution Module (DConv) + Large Convolution Kernel (LK)

### 2. Swin Spatial Pyramid Pooling (SSPP)
- **Purpose**: Extract long-distance dependencies within images and channel relationships
- **Innovation**: Swin Transformer blocks with cross-channel attention
- **Advantage**: Global and local feature integration with hierarchical processing
- **Design**: Multi-scale channel information fusion using MLP and GAP

### 3. Combined Attention Module (CAM)
- **Purpose**: Enhanced decoder performance through dual attention
- **Innovation**: DCT orthogonality for channel weighting and convolutional element-wise multiplication for spatial weighting
- **Components**:
  - **Channel Attention**: DCT-based orthogonal channel weighting
  - **Spatial Attention**: Element-wise multiplication for spatial focus
- **Advantage**: Improved generalization and fine edge detail preservation

### 4. Attention Gates (AG)
- **Purpose**: Highlight foreground while suppressing irrelevant background information
- **Location**: Skip connections between encoder and decoder
- **Function**: Adaptive feature selection and noise reduction

## News and Achievements

- **24-12-08**: Paper published on arXiv ([2412.06088](https://arxiv.org/abs/2412.06088))
- **24-08-20**: Paper accepted by **IEEE BIBM 2024**
- **State-of-the-Art**: **94.4% Dice score** on BraTS 2020 dataset
- **Multi-Benchmark**: Evaluated on 3 authoritative datasets + proprietary data

## Installation

### Requirements
```bash
Python 3.8+
PyTorch 1.8+
```

### Setup
```bash
git clone https://github.com/WendyWAAAAANG/A4-Unet.git
cd A4-Unet
pip install -r requirements.txt
```

## Project Structure

```
A4-Unet/
├── a4unet/
│   ├── dataloader/
│   │   ├── bratsloader.py         # BraTS dataset loader
│   │   ├── hippoloader.py         # Hippocampus dataset loader  
│   │   ├── isicloader.py          # ISIC skin lesion loader
│   │   └── preprocessor.py        # Data preprocessing utilities
│   └── model/
│       ├── a4unet.py              # Main A4-Unet model implementation
│       ├── D_LKA/
│       │   └── deformable_LKA.py  # Deformable Large Kernel Attention
│       ├── fca.py                 # Frequency Channel Attention
│       ├── fp16_util.py           # Half-precision utilities
│       ├── grid_attention.py      # Grid attention mechanisms
│       ├── logger.py              # Training logging utilities
│       ├── lr_scheduler.py        # Learning rate scheduling
│       ├── nn.py                  # Neural network utilities
│       ├── other.py               # Additional utilities
│       ├── pyramid_block.py       # Pyramid pooling blocks
│       ├── sspp_utils/            # Swin Spatial Pyramid Pooling
│       │   ├── config.py          # SSPP configuration
│       │   ├── cross_attn.py      # Cross-channel attention
│       │   ├── swin_224_7_2level.py # Swin Transformer variants
│       │   └── swin.py            # Main Swin Transformer
│       ├── sspp.py                # SSPP implementation
│       ├── swin_deeplab.py        # Swin-DeepLab integration
│       ├── unet_parts.py          # UNet building blocks
│       ├── unet.py                # Standard UNet implementation
│       └── utils.py               # General utilities
├── evaluate.py                    # Comprehensive model evaluation
├── predict.py                     # Model inference script
├── result.py                      # Results analysis and visualization
├── train.py                       # Training script
├── visualize.ipynb                # Jupyter notebook for visualization
├── requirement.txt                # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

## Usage

### Dataset Preparation

#### BraTS Dataset (Recommended)
1. Download BraTS 2020/2021 from [official source](https://www.med.upenn.edu/cbica/brats2020/data.html)
2. Organize structure:
```
data/
└── MICCAI_BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_t1.nii.gz      # T1-weighted
    │   ├── BraTS20_Training_001_t1ce.nii.gz    # T1-contrast enhanced
    │   ├── BraTS20_Training_001_t2.nii.gz      # T2-weighted  
    │   ├── BraTS20_Training_001_flair.nii.gz   # FLAIR
    │   └── BraTS20_Training_001_seg.nii.gz     # Segmentation mask
    └── BraTS20_Training_002/...
```

### Training

#### A4-Unet Training (Recommended)
```bash
# Basic A4-Unet training on BraTS
python train.py --a4unet --datasets Brats --input_size 128 --batch_size 16 --epochs 50

# Advanced training with optimization
python train.py \
    --a4unet \
    --datasets Brats \
    --input_size 128 \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 2e-4 \
    --validation 20 \
    --valstep 5 \
    --amp
```

#### Baseline UNet Training
```bash
python train.py --datasets Brats --input_size 128 --batch_size 16 --epochs 50
```

#### Training Parameters
- `--a4unet`: Enable A4-Unet architecture (vs standard UNet)
- `--datasets`: Dataset choice (`Brats`, `ISIC`, `Hippo`)  
- `--input_size`: Input resolution (128 for medical images)
- `--batch_size`: Training batch size (16 recommended)
- `--epochs`: Training epochs (50-100 for convergence)
- `--learning_rate`: AdamW learning rate (2e-4 for A4-Unet)
- `--amp`: Automatic mixed precision (memory efficiency)

### Evaluation

```bash
# Comprehensive evaluation with all metrics
python evaluate.py \
    --load ./checkpoints/checkpoint_epoch50.pth \
    --dataset Brats \
    --a4unet \
    --input_size 128 \
    --final_test
```

**Evaluation Metrics:**
- **Dice Score**: Overlap-based segmentation accuracy
- **mIoU**: Mean Intersection over Union  
- **HD95**: 95th percentile Hausdorff Distance (boundary accuracy)
- **Visualization**: Saved prediction masks

### Inference and Prediction

```bash
# Single image prediction
python predict.py \
    --load ./checkpoints/checkpoint_epoch50.pth \
    --input ./data/test_image.nii.gz \
    --output ./results/prediction.nii.gz

# Batch prediction on dataset
python predict.py \
    --load ./checkpoints/checkpoint_epoch50.pth \
    --dataset Brats \
    --input_dir ./data/test/ \
    --output_dir ./results/
```

### Results Analysis

```bash
# Analyze and visualize results
python result.py \
    --predictions ./results/ \
    --ground_truth ./data/masks/ \
    --output ./analysis/

# Interactive visualization (Jupyter notebook)
jupyter notebook visualize.ipynb
```

## Performance Results

### BraTS 2020 Dataset Benchmarks

| Method | Dice Score | mIoU | HD95 | Parameters |
|--------|------------|------|------|------------|
| UNet | 0.847 | 0.762 | 12.3 | 31.0M |
| **A4-Unet** | **0.944** | **0.891** | **8.7** | 28.5M |

### Key Performance Highlights
- **+9.7% Dice improvement** over standard UNet
- **+12.9% mIoU improvement** with better boundary detection
- **-29% HD95 reduction** indicating superior spatial accuracy
- **Fewer parameters** while achieving better performance

## Model Architecture Details

### A4-Unet Components

```python
# Import A4-Unet model
from a4unet.model.a4unet import create_a4unet_model

# Create A4-Unet model
model = create_a4unet_model(
    image_size=128,         # Input resolution
    num_channels=128,       # Feature channels
    num_res_blocks=2,       # Residual blocks
    num_classes=2,          # Background + tumor
    learn_sigma=True,       # Uncertainty estimation
    in_ch=4                 # T1, T1ce, T2, FLAIR
)
```

### Individual Component Access
```python
# Import specific components
from a4unet.model.D_LKA.deformable_LKA import DeformableLKA
from a4unet.model.sspp import SSPP
from a4unet.model.fca import FCA
from a4unet.model.grid_attention import GridAttention
```

### Architecture Specifications
- **Input**: 4-channel MRI (T1, T1ce, T2, FLAIR)
- **Encoder**: DLKA blocks with deformable convolutions
- **Bottleneck**: SSPP with Swin Transformer attention
- **Decoder**: CAM with DCT-based channel + spatial attention
- **Skip Connections**: Attention Gates for feature selection
- **Output**: 2-class segmentation (background, tumor)

## Advanced Features

### Multi-Scale Information Processing
- **Large Kernel Convolutions**: Capture diverse tumor sizes
- **Deformable Convolutions**: Adapt to irregular tumor shapes
- **Pyramid Pooling**: Multi-scale context aggregation

### Attention Mechanisms
- **Channel Attention**: DCT orthogonality for frequency domain weighting
- **Spatial Attention**: Element-wise multiplication for region focus
- **Cross-Channel Attention**: Inter-modality relationship modeling

### Medical Image Optimizations
- **4D MRI Support**: Multi-modal brain imaging
- **Slice-based Processing**: 3D volume → 2D slice training
- **Uncertainty Quantification**: Learn_sigma for confidence estimation

## Model Checkpoints

Pre-trained models available at: **https://huggingface.co/Roxanne-WANG/A4-Unet/tree/main**

### Loading Pre-trained Weights
```python
import torch
from a4unet.a4unet import create_a4unet_model

# Load model
model = create_a4unet_model(
    image_size=128, num_channels=128, num_res_blocks=2,
    num_classes=2, learn_sigma=True, in_ch=4
)

# Load checkpoint
checkpoint = torch.load('checkpoint_epoch50.pth', weights_only=True)
if 'mask_values' in checkpoint:
    del checkpoint['mask_values']
model.load_state_dict(checkpoint)
```

## Multi-Dataset Support

| Dataset | Task | Modalities | Classes | Resolution | Loader |
|---------|------|------------|---------|------------|---------|
| **BraTS** | Brain Tumor | 4 (T1,T1ce,T2,FLAIR) | 2 | 128×128 | `bratsloader.py` |
| **ISIC** | Skin Lesion | 3 (RGB) | 2 | 256×256 | `isicloader.py` |
| **Hippocampus** | Brain Structure | 4 (MRI) | 2 | 128×128 | `hippoloader.py` |
| **Tongue** | Tongue Segmentation | 3 (RGB) | 2 | 256×256 | `tongueloader.py` |
| **Tongue 2D** | Tongue 2D Slices | 3 (RGB) | 2 | 256×256 | `tongue2dloader.py` |

### Data Loading Options

```python
# Standard dataloaders (a4unet/dataloader/)
from a4unet.dataloader.bratsloader import BRATSDataset3D
from a4unet.dataloader.isicloader import ISICDataset
from a4unet.dataloader.hippoloader import HIPPODataset3D

# Advanced dataloaders (a4unet/model/dataloader/) 
from a4unet.model.dataloader.bratsloader import BRATSDataset3D
from a4unet.model.dataloader.loader import GenericDataLoader
from a4unet.model.dataloader.preprocessor import MedicalImagePreprocessor
```

## Citation

If you use A4-Unet in your research, please cite our IEEE BIBM 2024 paper:

```bibtex
@inproceedings{wang2024a4,
  title={A4-Unet: Deformable Multi-Scale Attention Network for Brain Tumor Segmentation},
  author={Wang, Ruoxin and Tang, Tianyi and Du, Haiming and Cheng, Yuxuan and Wang, Yu and Yang, Lingjie and Duan, Xiaohui and Yu, Yunfang and Zhou, Yu and Chen, Donglong},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={2583--2590},
  year={2024},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the License.md file for details.

## Ethical Statement

This research adheres to medical AI ethical guidelines:

1. **Data Compliance**: Uses publicly available datasets (BraTS, ISIC) with proper licensing
2. **Reproducibility**: Complete implementation and experimental settings provided  
3. **Medical Responsibility**: Intended for research advancement, not clinical diagnosis
4. **Open Science**: Code and model weights openly available for scientific validation

## Contact and Support

**Authors:** 
- Ruoxin Wang (ruoxinwaaang@gmail.com) - BNU-HKBU United International College, Duke University
- Tianyi Tang (trumantytang@163.com) - BNU-HKBU United International College, University of Illinois Urbana-Champaign
- Haiming Du (jennyduuu@163.com) - BNU-HKBU United International College, Rice University

**Corresponding Author:** Donglong Chen (donglongchen@uic.edu.cn)
