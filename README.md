# MedHyperMamba

This repository contains the official implementation of the paper:

**"MedHyperMamba: A Multi-Capable Mamba with Multimodal Registration, Denoising and Novel Hypergraph Scanning for MRI Brain Tumor Segmentation"**

## Requirements
- Python 3.8+
- PyTorch 2.1.0
- causal-conv1d 1.1.1
- mamba-ssm 2.2.2

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
