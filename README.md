# Lensless Camera Reconstruction

Implementation of ADMM-based algorithms for lensless computational imaging, based on:

- [Le-ADMM (Monakhova et al., 2019)](https://arxiv.org/abs/1908.11502) — Learned ADMM for lensless imaging
- [Modular Le-ADMM (Bezzam et al., 2025)](https://arxiv.org/abs/2502.01102) — Modular architecture with pre/post-processors

## Models

| Model | Description | Parameters |
|-------|-------------|------------|
| ADMM-100 | Standard ADMM with fixed hyperparameters (μ=1e-4, τ=2e-4) | 0 |
| Le-ADMM-20 | Unrolled ADMM with learnable μ_i, τ_i per iteration | 40 |
| Modular pre+post | Le-ADMM-5 + U-Net pre + U-Net post | ~8M |
| Modular pre only | Le-ADMM-5 + U-Net pre | ~8M |
| Modular post only | Le-ADMM-5 + U-Net post | ~8M |

## Installation

```bash
# Clone the repository
git clone https://github.com/USERNAME/lenless.git
cd lenless

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The project uses [DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K) from HuggingFace. The dataset is downloaded automatically during training.

## Training

Train a model using Hydra configs:

```bash
# Le-ADMM-20
python train.py --config-name=lensless model=le_admm20 writer.run_name=le_admm20

# Modular pre+post
python train.py --config-name=lensless model=modular_pre_post writer.run_name=modular_pre_post

# Modular pre only
python train.py --config-name=lensless model=modular_pre writer.run_name=modular_pre

# Modular post only
python train.py --config-name=lensless model=modular_post writer.run_name=modular_post
```

Training logs are saved to [Comet ML](https://www.comet.com/).

## Inference

Run inference on a dataset:

```bash
# On DigiCam test split
python inference.py \
    model=modular_pre_post \
    datasets=digicam_eval \
    inferencer.from_pretrained=saved/model_best.pth

# On custom directory
python inference.py \
    model=modular_pre_post \
    datasets=custom_dir \
    datasets.test.data_dir=/path/to/dataset \
    inferencer.from_pretrained=saved/model_best.pth
```

Custom directory format:
```
data_dir/
├── lensless/   *.png
├── masks/      *.npy
└── lensed/     *.png  (optional, ground truth)
```

## Metrics

Calculate metrics between ground truth and reconstructions:

```bash
python calculate_metrics.py --gt_dir /path/to/lensed --pred_dir /path/to/reconstructions
```

Metrics: PSNR, SSIM, LPIPS (VGG), MSE.

## Speed Benchmark

Measure inference speed for all models:

```bash
python calculate_speed.py --device cuda
```

## Demo

See `demo.ipynb` for a complete demonstration of:
- Repository setup and installation
- Checkpoint download
- Inference on custom dataset
- Visualization of results
- Metrics calculation

## Checkpoint Download

Download trained model checkpoints:

```bash
python scripts/download_checkpoints.py
```

## Results

| Model | PSNR | SSIM | LPIPS | MSE |
|-------|------|------|-------|-----|
| ADMM-100 | TBD | TBD | TBD | TBD |
| Le-ADMM-20 | TBD | TBD | TBD | TBD |
| Modular pre+post | TBD | TBD | TBD | TBD |
| Modular pre only | TBD | TBD | TBD | TBD |
| Modular post only | TBD | TBD | TBD | TBD |

## Project Structure

```
lenless/
├── src/
│   ├── configs/          # Hydra configs
│   ├── datasets/         # Dataset implementations
│   ├── loss/             # Loss functions
│   ├── metrics/          # Metrics (PSNR, SSIM, LPIPS, MSE)
│   ├── model/            # Model implementations
│   │   ├── admm.py       # Standard ADMM
│   │   ├── le_admm.py    # Unrolled Le-ADMM
│   │   ├── modular_le_admm.py  # Modular Le-ADMM
│   │   ├── unet.py       # U-Net for processors
│   │   └── fft_utils.py  # FFT utilities
│   ├── trainer/          # Training and inference
│   └── utils/            # Utilities
├── train.py              # Training script
├── inference.py          # Inference script
├── calculate_metrics.py  # Metrics calculation
├── calculate_speed.py    # Speed benchmark
├── demo.ipynb            # Demo notebook
└── requirements.txt      # Dependencies
```

## License

MIT License
