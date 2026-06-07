"""
Standard ADMM for lensless image reconstruction with TV regularization.
Fixed hyperparameters (not learnable).
"""

import torch
from torch import nn

from src.model.fft_utils import (
    compute_d_otf,
    compute_fft_shape,
    crop_from_fft,
    finite_diff,
    pad_to_fft,
    psf_to_otf,
    soft_threshold,
)


class ADMM(nn.Module):
    """
    Standard ADMM for lensless image reconstruction with TV regularization.
    Fixed hyperparameters (not learnable).
    """

    def __init__(self, n_iter=100, mu=1e-4, tau=2e-4):
        super().__init__()
        self.n_iter = n_iter
        self.mu = mu
        self.tau = tau

    def forward(self, lensless, mask, **batch):
        """
        Args:
            lensless: (B, C, H, W) lensless measurement
            mask: (B, H_m, W_m) PSF/mask
        Returns:
            dict with "reconstruction": (B, C, H, W) reconstructed image
        """
        B, C, H, W = lensless.shape
        device = lensless.device

        # Compute FFT shape (padded space)
        fft_shape = compute_fft_shape(H, W)

        # Prepare mask: add channel dim if needed → (B, 1, H_m, W_m)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        # Expand mask to match channels if single-channel
        # PSF is the same for all RGB channels

        # Compute OTF from PSF
        otf = psf_to_otf(mask, fft_shape)  # (B, 1, fft_H, fft_W//2+1)
        otf_conj = torch.conj(otf)
        otf_abs_sq = torch.abs(otf) ** 2  # |H|^2

        # Compute finite difference OTFs
        dx_otf, dy_otf = compute_d_otf(fft_shape, device)
        dx_abs_sq = torch.abs(dx_otf) ** 2
        dy_abs_sq = torch.abs(dy_otf) ** 2

        # Precompute denominator for x-update (fixed mu)
        # denom = |H|^2 + mu * (|Dx|^2 + |Dy|^2)
        denom = otf_abs_sq + self.mu * (dx_abs_sq + dy_abs_sq) + 1e-8

        # Precompute A^T y in frequency domain
        y_padded = pad_to_fft(lensless, fft_shape)
        Y = torch.fft.rfft2(y_padded)
        ATy = otf_conj * Y  # (B, C, fft_H, fft_W//2+1)

        # Initialize variables (all zeros in padded space)
        # z = (zx, zy) — gradient-domain dual variables
        zx = torch.zeros(B, C, fft_shape[0], fft_shape[1], device=device)
        zy = torch.zeros_like(zx)
        ux = torch.zeros_like(zx)
        uy = torch.zeros_like(zx)

        for _ in range(self.n_iter):
            # x-update in frequency domain
            # rhs = A^T y + mu * D^T(z - u)
            rhs_spatial = finite_diff_adjoint_from_components(
                zx - ux, zy - uy
            )
            RHS = ATy + self.mu * torch.fft.rfft2(rhs_spatial)
            X = RHS / denom
            x = torch.fft.irfft2(X, s=fft_shape)

            # Compute Dx for z-update and u-update
            dx, dy = finite_diff(x)

            # z-update: soft thresholding
            zx = soft_threshold(dx + ux, self.tau / self.mu)
            zy = soft_threshold(dy + uy, self.tau / self.mu)

            # u-update
            ux = ux + dx - zx
            uy = uy + dy - zy

        # Crop to original size and clamp
        reconstruction = crop_from_fft(x, (H, W))
        reconstruction = torch.clamp(reconstruction, 0.0, 1.0)

        return {"reconstruction": reconstruction}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info


def finite_diff_adjoint_from_components(dx, dy):
    """Helper: compute D^T from separate dx, dy components.
    
    Adjoint of Dx[i,j] = x[i,j+1] - x[i,j] is:
    D^T_x v[i,j] = v[i,j-1] - v[i,j] = roll(v, +1) - v
    """
    adj_dx = torch.roll(dx, shifts=1, dims=-1) - dx
    adj_dy = torch.roll(dy, shifts=1, dims=-2) - dy
    return adj_dx + adj_dy
