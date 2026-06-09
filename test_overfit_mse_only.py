"""
One-batch overfit test for Le-ADMM with MSE loss only (no LPIPS).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from src.model.le_admm import LeADMM

print("=" * 60)
print("One-batch overfit test for Le-ADMM (MSE only, no LPIPS)")
print("=" * 60)

# Create synthetic data: one batch
B, C, H, W = 1, 3, 64, 64

# Random lensless measurement
lensless = torch.rand(B, C, H, W)

# Random PSF/mask (normalized)
mask = torch.rand(B, H, W)
mask = mask / mask.sum(dim=(-2, -1), keepdim=True)

# Random ground truth
lensed = torch.rand(B, C, H, W)

print(f"Batch shapes:")
print(f"  lensless: {lensless.shape}")
print(f"  mask: {mask.shape}")
print(f"  lensed: {lensed.shape}")

# Create model
model = LeADMM(n_iter=5, init_mu=1e-4, init_tau=2e-4)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1.0)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

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

if loss_reduction > 50:
    print("✓ PASS: Model successfully overfitted the batch (loss reduced > 50%)")
else:
    print("✗ FAIL: Model did NOT overfit (loss reduction < 50%)")

print()
print("=" * 60)
