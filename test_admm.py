"""
Standalone test for ADMM algorithm.
Tests the forward model and ADMM reconstruction on synthetic data.
No GPU required.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

print("=" * 60)
print("Testing FFT utilities...")
print("=" * 60)

from src.model.fft_utils import (
    compute_fft_shape,
    pad_to_fft,
    crop_from_fft,
    psf_to_otf,
    compute_d_otf,
    finite_diff,
    finite_diff_adjoint,
    soft_threshold,
)

# Test 1: compute_fft_shape
H, W = 270, 480
fft_H, fft_W = compute_fft_shape(H, W)
print(f"[OK] compute_fft_shape({H}, {W}) -> ({fft_H}, {fft_W})")
assert fft_H >= 2 * H and fft_W >= 2 * W, "FFT shape too small"

# Test 2: pad_to_fft and crop_from_fft roundtrip
x = torch.rand(2, 3, H, W)
x_padded = pad_to_fft(x, (fft_H, fft_W))
assert x_padded.shape == (2, 3, fft_H, fft_W), f"Wrong padded shape: {x_padded.shape}"
x_cropped = crop_from_fft(x_padded, (H, W))
assert x_cropped.shape == (2, 3, H, W), f"Wrong cropped shape: {x_cropped.shape}"
# The center region should match original
print(f"[OK] pad_to_fft -> {x_padded.shape}, crop_from_fft -> {x_cropped.shape}")
# Check that crop recovers original
assert torch.allclose(x_cropped, x, atol=1e-6), "Pad/crop roundtrip failed!"
print("[OK] pad/crop roundtrip: original == cropped")

# Test 3: psf_to_otf
psf = torch.rand(2, 1, H, W)
psf = psf / psf.sum(dim=(-2, -1), keepdim=True)  # normalize
otf = psf_to_otf(psf, (fft_H, fft_W))
assert otf.shape == (2, 1, fft_H, fft_W // 2 + 1), f"Wrong OTF shape: {otf.shape}"
assert otf.is_complex(), "OTF should be complex"
print(f"[OK] psf_to_otf -> {otf.shape}, complex: {otf.is_complex()}")

# Test 4: compute_d_otf
dx_otf, dy_otf = compute_d_otf((fft_H, fft_W), device=torch.device("cpu"))
assert dx_otf.shape == (1, 1, fft_H, fft_W // 2 + 1)
assert dy_otf.shape == (1, 1, fft_H, fft_W // 2 + 1)
print(f"[OK] compute_d_otf -> dx: {dx_otf.shape}, dy: {dy_otf.shape}")

# Test 5: finite_diff and adjoint
x_test = torch.rand(1, 1, 8, 8)
dx, dy = finite_diff(x_test)
adj = finite_diff_adjoint(dx, dy)
assert dx.shape == x_test.shape
assert dy.shape == x_test.shape
assert adj.shape == x_test.shape
print(f"[OK] finite_diff and adjoint shapes correct")

# Test adjoint property: <Dx, v> == <x, D^T v>
v = torch.rand_like(x_test)
dv_x, dv_y = finite_diff(v)
lhs = (dx * v).sum() + (dy * v).sum()  # <Dx, v> (treating D as [Dx; Dy])
rhs = (x_test * finite_diff_adjoint(v, v)).sum()  # <x, D^T v>
# Note: adjoint test with same v for both components
# Proper test: <[dx;dy], [vx;vy]> = <x, D^T[vx;vy]>
vx = torch.rand_like(x_test)
vy = torch.rand_like(x_test)
dx2, dy2 = finite_diff(x_test)
lhs2 = (dx2 * vx).sum() + (dy2 * vy).sum()
rhs2 = (x_test * finite_diff_adjoint(vx, vy)).sum()
print(f"[OK] Adjoint test: <Dx,v>={lhs2.item():.6f}, <x,D^Tv>={rhs2.item():.6f}, diff={abs(lhs2.item()-rhs2.item()):.2e}")
assert abs(lhs2.item() - rhs2.item()) < 1e-4, "Adjoint property violated!"

# Test 6: soft_threshold
x_st = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
result = soft_threshold(x_st, 1.0)
expected = torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0])
assert torch.allclose(result, expected, atol=1e-6), f"Soft threshold wrong: {result}"
print(f"[OK] soft_threshold: {result.tolist()}")

print()
print("=" * 60)
print("Testing ADMM forward pass...")
print("=" * 60)

from src.model.admm import ADMM

# Create a small synthetic test
B, C, H_small, W_small = 1, 3, 64, 64

# Create a simple Gaussian PSF
def gaussian_psf(H, W, sigma=5.0):
    y = torch.arange(H).float() - H // 2
    x = torch.arange(W).float() - W // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    psf = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf = psf / psf.sum()
    return psf

psf = gaussian_psf(H_small, W_small)
mask = psf.unsqueeze(0)  # (1, H, W) — batch of 1

# Create a synthetic "ground truth" image
gt = torch.rand(B, C, H_small, W_small)

# Simulate lensless measurement: y = conv(h, x)
# Use the forward model manually
fft_shape = compute_fft_shape(H_small, W_small)
mask_4d = mask.unsqueeze(1)  # (1, 1, H, W)
otf = psf_to_otf(mask_4d, fft_shape)  # (1, 1, fft_H, fft_W//2+1)
gt_padded = pad_to_fft(gt, fft_shape)
GT = torch.fft.rfft2(gt_padded)
Y = otf * GT  # (1, 3, fft_H, fft_W//2+1) — broadcast over channels
y_padded = torch.fft.irfft2(Y, s=fft_shape)
lensless = crop_from_fft(y_padded, (H_small, W_small))
lensless = torch.clamp(lensless, 0.0, 1.0)

print(f"Synthetic data: gt={gt.shape}, lensless={lensless.shape}, mask={mask.shape}")

# Test ADMM with few iterations (fast)
model = ADMM(n_iter=10, mu=1e-4, tau=2e-4)
print(f"ADMM model: {model.n_iter} iterations, mu={model.mu}, tau={model.tau}")

with torch.no_grad():
    output = model(lensless, mask)

assert "reconstruction" in output, "Output must have 'reconstruction' key"
recon = output["reconstruction"]
assert recon.shape == (B, C, H_small, W_small), f"Wrong reconstruction shape: {recon.shape}"
assert recon.min() >= -1e-6, f"Reconstruction has negative values: {recon.min()}"
assert recon.max() <= 1.0 + 1e-6, f"Reconstruction exceeds 1: {recon.max()}"

# Compute MSE
mse = F.mse_loss(recon, gt).item()
print(f"[OK] ADMM reconstruction shape: {recon.shape}")
print(f"[OK] Reconstruction range: [{recon.min():.4f}, {recon.max():.4f}]")
print(f"[OK] MSE (10 iters): {mse:.6f}")

# Test with more iterations — should improve
model_100 = ADMM(n_iter=50, mu=1e-4, tau=2e-4)
with torch.no_grad():
    output_100 = model_100(lensless, mask)
recon_100 = output_100["reconstruction"]
mse_100 = F.mse_loss(recon_100, gt).item()
print(f"[OK] MSE (50 iters): {mse_100:.6f}")

# Test __str__
print(f"\nModel info:\n{model}")

print()
print("=" * 60)
print("Testing Le-ADMM forward pass...")
print("=" * 60)

from src.model.le_admm import LeADMM

model_le = LeADMM(n_iter=5, init_mu=1e-4, init_tau=2e-4)
print(f"Le-ADMM model: {model_le.n_iter} iterations")

with torch.no_grad():
    output_le = model_le(lensless, mask)

assert "reconstruction" in output_le, "Output must have 'reconstruction' key"
recon_le = output_le["reconstruction"]
assert recon_le.shape == (B, C, H_small, W_small), f"Wrong shape: {recon_le.shape}"
mse_le = F.mse_loss(recon_le, gt).item()
print(f"[OK] Le-ADMM reconstruction shape: {recon_le.shape}")
print(f"[OK] MSE (5 iters): {mse_le:.6f}")

# Count parameters
n_params = sum(p.numel() for p in model_le.parameters())
n_trainable = sum(p.numel() for p in model_le.parameters() if p.requires_grad)
print(f"[OK] Parameters: {n_params} total, {n_trainable} trainable (expected {2*5}=10)")
assert n_trainable == 2 * 5, f"Expected 10 trainable params, got {n_trainable}"

print()
print("=" * 60)
print("All tests PASSED!")
print("=" * 60)
