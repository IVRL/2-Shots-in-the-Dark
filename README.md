# 2-Shots in the Dark: Low-Light Denoising with Minimal Data Acquisition

This repository contains the official implementation of **"2-Shots in the Dark: Low-Light Denoising with Minimal Data Acquisition"**.

📄 [View on arXiv](https://arxiv.org/abs/2512.03245)

## Repository Structure

```
├── dataloader/            # Dataset classes for training and evaluation
├── demos/                 # Jupyter notebooks for noise analysis and dark frame synthesis
│   ├── generated_darkframe_demo.ipynb      # Dark frame generation demo
│   └── poisson_noise_analysis_demo.ipynb   # Poisson noise parameter estimation from a single noisy image
├── models/                # Network architectures and training logic
├── utils/                 # Utility functions (raw processing, metrics, etc.)
├── train_denoising.py     # Training script
├── train.sh               # Training shell commands for SID and LRID
├── test_denoising.py      # Testing script for SID and ELD datasets
├── test_LRID_denoising.py # Testing script for LRID dataset
├── test_sid.sh            # Shell script for SID/ELD evaluation
├── test_lrid.sh           # Shell script for LRID evaluation
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-compatible GPU

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Pretrained Weights and Resources

Pretrained model checkpoints and required resource files (dark shadings, sampled dark frames, etc.) are available on Google Drive:

**[Download pretrained weights and resources](https://drive.google.com/drive/folders/1Hz4rzV8578lHCTzQkwCHE2nrphQy1QYG?usp=sharing)**

After downloading, place the files as follows:

```
├── pretrained_ckpts/
│   ├── sid_eld_final.pth       # Checkpoint for SID/ELD evaluation
│   └── lrid_final.pth          # Checkpoint for LRID evaluation
├── data/
│   ├── SID_DarkShadings_BlurSigma50/
│   ├── SID_SampledDarkFrames/
│   ├── LRID_DarkShadings_BlurSigma50/
│   ├── LRID_Hot_DarkShadings_BlurSigma50/
│   ├── LRID_SampledDarkFrames/
│   └── LRID_Hot_SampledDarkFrames/
```

### Datasets

- [SID Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)
- [ELD Dataset](https://github.com/Vandermode/ELD)
- [LRID Dataset](https://github.com/megvii-research/PMN/tree/TPAMI)

Update the dataset paths in the test scripts or directly in `test_denoising.py` / `test_LRID_denoising.py` to match your local setup.

## Demos

We provide two Jupyter notebooks under `demos/` to illustrate key components of our method:

- **`generated_darkframe_demo.ipynb`** — Demonstrates how to synthesize dark frames from minimal acquisitions, modeling signal-independent noise.
- **`poisson_noise_analysis_demo.ipynb`** — Shows how to estimate the Poisson noise parameter (shot noise) from a single noisy raw image.

## Training

### SID Dataset

```bash
python train_denoising.py \
  --use_tb_logger --loss_l1 \
  --vis_freq 50 --save_epoch_freq 50 \
  --name train_denoising_sid \
  --trainset SIDSyntheticDataset \
  --batch_size 4 --max_iter 500 --crop_size 256 \
  --darkshading_folder data/SID_DarkShadings_BlurSigma50 \
  --darkframe_folder data/SID_SampledDarkFrames \
  --darkframe_num 400
```

### LRID Dataset

```bash
python train_denoising.py \
  --use_tb_logger --loss_l1 \
  --vis_freq 50 --save_epoch_freq 50 \
  --name train_denoising_lrid \
  --trainset LRIDSyntheticDataset \
  --batch_size 4 --max_iter 500 --crop_size 256 \
  --darkshading_folder data/LRID_DarkShadings_BlurSigma50 \
  --hot_darkshading_folder data/LRID_Hot_DarkShadings_BlurSigma50 \
  --darkframe_folder data/LRID_SampledDarkFrames \
  --hot_darkframe_folder data/LRID_Hot_SampledDarkFrames \
  --darkframe_num 400
```

Or simply run:

```bash
bash train.sh
```

Training logs and checkpoints are saved under `./logs/denoising/weights/<experiment_name>/`.

## Evaluation

### SID and ELD

```bash
# SID — ratios: 100, 250, 300
python test_denoising.py \
  --save_folder results/sid_eld \
  --resume pretrained_ckpts/sid_eld_final.pth \
  --test_dataset SID \
  --ratio 300 \
  --correct_darkshading --correct_illum --visualize_img

# ELD — ratios: 100, 200
python test_denoising.py \
  --save_folder results/sid_eld \
  --resume pretrained_ckpts/sid_eld_final.pth \
  --test_dataset ELD \
  --ratio 200 \
  --correct_darkshading --correct_illum --visualize_img
```

### LRID

```bash
# Indoor — ratios: 1, 2, 4, 8, 16
python test_LRID_denoising.py \
  --save_folder results/lrid \
  --resume pretrained_ckpts/lrid_final.pth \
  --condition indoor \
  --ratio 16 \
  --correct_illum --visualize_img

# Outdoor — ratios: 1, 2, 4
python test_LRID_denoising.py \
  --save_folder results/lrid \
  --resume pretrained_ckpts/lrid_final.pth \
  --condition outdoor \
  --ratio 4 \
  --correct_illum --visualize_img
```

To run the full evaluation across all datasets and ratios:

```bash
bash test_sid.sh   # Evaluates on SID and ELD
bash test_lrid.sh  # Evaluates on LRID (indoor + outdoor)
```

Results (PSNR/SSIM per image and averaged) are printed to the console and saved to text files in the output folder. When `--visualize_img` is enabled, denoised output images are also saved.

## TODO

- [ ] Upload synthetic dark frames

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lu2026twoshots,
  title={2-Shots in the Dark: Low-Light Denoising with Minimal Data Acquisition},
  author={Lu, Liying and Achddou, Rapha{\"e}l and S{\"u}sstrunk, Sabine},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Acknowledgements

This project builds upon the following works and datasets:
- [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark)
- [ELD](https://github.com/Vandermode/ELD)
- [PMN](https://github.com/megvii-research/PMN)


