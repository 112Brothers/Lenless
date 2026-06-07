"""
Measure inference speed for all model variants.
"""
import argparse
import time
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def measure_speed(model, lensless, mask, device, n_warmup=10, n_runs=100):
    """Measure inference time in ms per image."""
    model.eval()
    model.to(device)
    lensless = lensless.to(device)
    mask = mask.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(lensless, mask)

    # Measure
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(lensless, mask)
    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()

    avg_time_ms = (end - start) * 1000 / n_runs
    return avg_time_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create dummy input
    B, C, H, W = 1, 3, 270, 480  # Typical DigiCam image size
    lensless = torch.randn(B, C, H, W)
    mask = torch.randn(B, H, W)
    mask = mask / mask.sum()

    # Model configs to test
    model_configs = [
        ("ADMM-100", "src/configs/model/admm100.yaml"),
        ("Le-ADMM-20", "src/configs/model/le_admm20.yaml"),
        ("Modular pre+post", "src/configs/model/modular_pre_post.yaml"),
        ("Modular pre only", "src/configs/model/modular_pre.yaml"),
        ("Modular post only", "src/configs/model/modular_post.yaml"),
    ]

    print(f"\n{'Model':<20} {'Time (ms)':>12} {'FPS':>8}")
    print("-" * 42)

    for name, config_path in model_configs:
        config = OmegaConf.load(config_path)
        model = instantiate(config)

        avg_time_ms = measure_speed(model, lensless, mask, device)
        fps = 1000 / avg_time_ms

        print(f"{name:<20} {avg_time_ms:>12.2f} {fps:>8.1f}")


if __name__ == "__main__":
    main()
