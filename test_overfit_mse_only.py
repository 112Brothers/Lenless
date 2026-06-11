"""
One-batch overfit test for Le-ADMM with MSE loss only (no LPIPS).

IMPORTANT: Le-ADMM is a physics-based solver with only 10 scalar parameters
(log_mu, log_tau per iteration). It CANNOT overfit random data — it can only
tune regularization strength.

The correct test uses physically consistent data:
  lensless = crop(PSF * ground_truth)  (forward model)
Then checks if ADMM can recover ground_truth.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from src.model.le_admm import LeADMM
from src.model.fft_utils import (
    compute_fft_shape, pad_to_fft, crop_from_fft, psf_to_otf
)

print("=" * 60)
print("One-batch overfit test for Le-ADMM (MSE only, no LPIPS)")
print("Physically consistent data: lensless = forward_model(gt)")
print("=" * 60)

torch.manual_seed(42)

# Create synthetic data: one batch
B, C, H, W = 1, 3, 64, 64

# Random ground truth image
ground_truth = torch.rand(B, C, H, W)

# Random PSF/mask (normalized)
mask = torch.rand(B, H, W)
mask = mask / mask.sum(dim=(-2, -1), keepdim=True)

# Simulate lensless measurement: y = crop(PSF * x)
# This is the forward model that ADMM inverts
with torch.no_grad():
    fft_shape = compute_fft_shape(H, W)
    mask_4d = mask.unsqueeze(1)  # (B, 1, H, W)
    otf = psf_to_otf(mask_4d, fft_shape)  # (B, 1, fft_H, fft_W//2+1)

    x_padded = pad_to_fft(ground_truth, fft_shape)
    X = torch.fft.rfft2(x_padded)
    Y = otf * X  # convolution in frequency domain
    y_full = torch.fft.irfft2(Y, s=fft_shape)
    lensless = crop_from_fft(y_full, (H, W))  # crop to measurement size

    # Normalize lensless to [0, 1]
    lensless = lensless - lensless.min()
    lensless = lensless / (lensless.max() + 1e-8)

# The target is the ground truth
lensed = ground_truth

print(f"Batch shapes:")
print(f"  lensless: {lensless.shape}")
print(f"  mask: {mask.shape}")
print(f"  lensed: {lensed.shape}")
print(f"  lensless range: [{lensless.min():.3f}, {lensless.max():.3f}]")

# Create model
model = LeADMM(n_iter=5, init_mu=1e-4, init_tau=2e-4)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Initial tau/mu = {2e-4/1e-4:.4f}")

# Training loop
n_epochs = 200
losses = []

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Forward pass
    output = model(lensless, mask)
    reconstruction = output["reconstruction"]

    # Compute loss (MSE only)
    loss = mse_loss(reconstruction, lensed)

    # Backward pass
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: loss={loss.item():.6f}")

print()
print("=" * 60)
print("Overfit test results:")
print("=" * 60)

# Check if loss decreased significantly
initial_loss = losses[0]
final_loss = losses[-1]
loss_reduction = (initial_loss - final_loss) / initial_loss * 100

print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss:   {final_loss:.6f}")
print(f"Reduction:    {loss_reduction:.1f}%")

# Print learned parameter values
print("\nLearned parameters (first/last iteration):")
mu0 = torch.exp(model.log_mu[0]).item()
tau0 = torch.exp(model.log_tau[0]).item()
mu_last = torch.exp(model.log_mu[-1]).item()
tau_last = torch.exp(model.log_tau[-1]).item()
print(f"  iter 0: mu={mu0:.2e}, tau={tau0:.2e}, tau/mu={tau0/mu0:.4f}")
print(f"  iter N: mu={mu_last:.2e}, tau={tau_last:.2e}, tau/mu={tau_last/mu_last:.4f}")

# For physics-based model with consistent data, expect >10% reduction
# (ADMM can tune regularization to better recover the signal)
if loss_reduction > 10:
    print("✓ PASS: Model improved reconstruction on physically consistent data")
else:
    print("✗ FAIL: Model did NOT improve (check gradient flow or forward model)")
    print("  Note: Le-ADMM cannot overfit RANDOM data — only physically consistent data.")

print()
print("=" * 60)
