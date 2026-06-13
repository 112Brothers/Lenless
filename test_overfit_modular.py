"""
One-batch overfit test for ModularLeADMM (pre+post DRUNet).

Uses small synthetic data to verify:
1. Forward pass works
2. Loss decreases (gradient flows through all components)
3. Output stays in [0, 1]

Run with: python test_overfit_modular.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim

from src.model.modular_le_admm import ModularLeADMM
from src.loss.reconstruction_loss import ReconstructionLoss

print("=" * 60)
print("One-batch overfit test for ModularLeADMM")
print("=" * 60)

# Use small image to keep it fast (no GPU needed)
B, C, H, W = 1, 3, 80, 135  # ~half of 160x270

# Random lensless measurement in [0, 1]
torch.manual_seed(42)
lensless = torch.rand(B, C, H, W)

# Small PSF mask (proportional to image: 54*scale x 26*scale, scale~0.53)
mask_h, mask_w = 29, 14
mask = torch.rand(B, mask_h, mask_w)
mask = mask / mask.sum(dim=(-2, -1), keepdim=True)

# Random ground truth in [0, 1]
lensed = torch.rand(B, C, H, W)

print(f"Batch shapes:")
print(f"  lensless: {lensless.shape}")
print(f"  mask:     {mask.shape}")
print(f"  lensed:   {lensed.shape}")

# Create model with small channels for speed
model = ModularLeADMM(
    n_iter=5,
    init_mu=1e-4,
    init_tau=2e-4,
    use_pre=True,
    use_post=True,
    drunet_channels=(16, 32, 64, 128),  # smaller for speed
    drunet_n_res_blocks=2,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# MSE-only loss for speed (no LPIPS)
criterion = ReconstructionLoss(lpips_weight=0.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"\nTraining for 50 steps...")
losses = []

for step in range(50):
    optimizer.zero_grad()

    output = model(lensless=lensless, mask=mask)
    reconstruction = output["reconstruction"]

    loss_dict = criterion(reconstruction=reconstruction, lensed=lensed)
    loss = loss_dict["loss"]

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())

    if (step + 1) % 10 == 0:
        rmin = reconstruction.min().item()
        rmax = reconstruction.max().item()
        print(f"  Step {step+1:3d}: loss={loss.item():.6f}, "
              f"recon=[{rmin:.3f},{rmax:.3f}]")

print()
print("=" * 60)
print("Results:")
print("=" * 60)

initial_loss = losses[0]
final_loss = losses[-1]
loss_reduction = (initial_loss - final_loss) / (initial_loss + 1e-8) * 100

print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss:   {final_loss:.6f}")
print(f"Reduction:    {loss_reduction:.1f}%")

# Check output range
with torch.no_grad():
    out = model(lensless=lensless, mask=mask)
    recon = out["reconstruction"]
    print(f"Output range: [{recon.min():.4f}, {recon.max():.4f}]  (should be [0,1])")
    in_range = (recon.min() >= 0.0) and (recon.max() <= 1.0)
    print(f"Output in [0,1]: {in_range}")

print()
if loss_reduction > 20:
    print("✓ PASS: ModularLeADMM overfits (loss reduced > 20%)")
else:
    print("✗ FAIL: ModularLeADMM did NOT overfit (loss reduction < 20%)")
    print("  Check: gradient flow, model architecture, loss function")

# Check gradients for all parameter groups
print()
print("Gradient check:")
optimizer.zero_grad()
out = model(lensless=lensless, mask=mask)
loss_dict = criterion(reconstruction=out["reconstruction"], lensed=lensed)
loss_dict["loss"].backward()

groups = {
    "pre_processor": model.pre_processor if model.use_pre else None,
    "le_admm.log_tau": model.le_admm.log_tau,
    "le_admm.log_mu1": model.le_admm.log_mu1,
    "post_processor": model.post_processor if model.use_post else None,
}

for name, params in groups.items():
    if params is None:
        continue
    if hasattr(params, 'parameters'):
        grads = [p.grad for p in params.parameters() if p.grad is not None]
        has_grad = len(grads) > 0
        if has_grad:
            avg_grad = sum(g.abs().mean().item() for g in grads) / len(grads)
            print(f"  {name}: grad OK (avg |grad|={avg_grad:.2e})")
        else:
            print(f"  {name}: NO GRADIENT!")
    else:
        # ParameterList
        grads = [p.grad for p in params if p.grad is not None]
        has_grad = len(grads) > 0
        if has_grad:
            avg_grad = sum(g.abs().mean().item() for g in grads) / len(grads)
            print(f"  {name}: grad OK (avg |grad|={avg_grad:.2e})")
        else:
            print(f"  {name}: NO GRADIENT!")

print()
print("=" * 60)
