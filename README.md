# MedHyperMamba

This repository contains the official implementation of the paper:

**"MedHyperMamba: A Multi-Capable Mamba with Multimodal Registration, Denoising and Novel Hypergraph Scanning for MRI Brain Tumor Segmentation"** (accepted to Biomedical Signal Processing and Control)

## Requirements
- Python 3.8+
- PyTorch 1.12+
- MONAI
- ...

## Installation
```bash
git clone https://github.com/acaneyoru/MedHyperMamba.git
cd MedHyperMamba
pip install -r requirements.txt
```

## Data Preparation
Download BraTS 2019/2020/2021 datasets and organize as follows:
```
data/
├── BraTS2019/
├── BraTS2020/
└── BraTS2021/
```

## Training
```bash
python train.py --dataset BraTS2019 --epochs 300 --batch_size 4
```

## Testing
```bash
python test.py --checkpoint /path/to/checkpoint
```

## Citation
If you find this code useful, please cite our paper:
```
@article{...}
```
