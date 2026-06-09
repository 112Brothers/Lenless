"""
Debug: check if gradients flow through Le-ADMM parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.model.le_admm import LeADMM
from src.loss.reconstruction_loss import ReconstructionLoss

print("=" * 60)
print("Gradient flow test for Le-ADMM")
print("=" * 60)

# Create synthetic data
B, C, H, W = 1, 3, 64, 64
lensless = torch.rand(B, C, H, W, requires_grad=False)
mask = torch.rand(B, H, W)
mask = mask / mask.sum(dim=(-2, -1), keepdim=True)
lensed = torch.rand(B, C, H, W, requires_grad=False)

# Create model
model = LeADMM(n_iter=5, init_mu=1e-4, init_tau=2e-4)
criterion = ReconstructionLoss(lpips_weight=0.1)

print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# Forward pass
output = model(lensless, mask)
reconstruction = output["reconstruction"]

print(f"\nReconstruction shape: {reconstruction.shape}")
print(f"Reconstruction range: [{reconstruction.min():.4f}, {reconstruction.max():.4f}]")

# Compute loss
loss_dict = criterion(reconstruction=reconstruction, lensed=lensed)
loss = loss_dict["loss"]

print(f"\nLoss: {loss.item():.6f}")
print(f"Loss requires_grad: {loss.requires_grad}")

# Backward pass
loss.backward()

print("\nGradients after backward:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name}: grad_norm={grad_norm:.2e}, grad_max={param.grad.abs().max().item():.2e}")
    else:
        print(f"  {name}: NO GRADIENT")

print()
print("=" * 60)
