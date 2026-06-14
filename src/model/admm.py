"""
Standard ADMM (fixed hyperparameters) for lensless reconstruction.

Reference: "Learned reconstructions for practical mask-based lensless imaging"
           Monakhova et al., Opt. Express 2019. https://arxiv.org/abs/1908.11502

ADMM solves: min_x 0.5*||C*H*x - y||^2 + tau*TV(x)  s.t. x >= 0
via 3-variable splitting:
  v = C*H*x  (data fidelity, crop operator C)
  u = D*x    (TV, finite differences D)
  w = x      (non-negativity)

x-update denominator: mu1*|H|^2 + mu2*(|Dx|^2 + |Dy|^2) + mu3
u-update: soft_threshold(D*x + alpha2/mu2, tau/mu2)
v-update: (mu1*C*H*x + Cty + alpha1) / (mu1 + 1)
w-update: max(x + alpha3/mu3, 0)

Hyperparameters per task spec: mu1=mu2=mu3=1e-4, tau=2e-4.
TV regularizer: anisotropic (separate x/y soft-thresholding).
Finite differences: circular (roll-based).
Working space: padded ~2H x 2W FFT-friendly space.
"""

import torch
from torch import nn

from src.model.fft_utils import (
    compute_d_otf,
    compute_fft_shape,
    crop_from_fft,
    finite_diff,
    finite_diff_adjoint,
    pad_to_fft,
    psf_to_otf,
    soft_threshold,
)


class ADMM(nn.Module):
    """
    Standard ADMM with fixed hyperparameters for lensless reconstruction.

    3-variable splitting ADMM:
      v = C*H*x  (data fidelity with crop C)
      u = D*x    (TV regularization)
      w = x      (non-negativity constraint)
    """

    def __init__(self, n_iter=100, mu=1e-4, tau=2e-4):
        super().__init__()
        self.n_iter = n_iter
        self.mu = mu    # penalty for all three constraints (mu1=mu2=mu3=mu)
        self.tau = tau  # TV soft-threshold parameter
        self.psf_scale = 5.0  # fixed scale matching empirical findings

    def forward(self, lensless, mask, **batch):
        """
        Args:
            lensless: (B, C, H, W) lensless measurement in [0, 1]
            mask: (B, H_m, W_m) PSF/mask (normalized to sum=1)
        Returns:
            dict with "reconstruction": (B, C, H, W) in [0, 1]
        """
        B, C, H, W = lensless.shape
        device = lensless.device
        mu1 = mu2 = mu3 = self.mu
        tau = self.tau

        # Compute padded FFT shape (~2H x 2W, FFT-friendly)
        fft_shape = compute_fft_shape(H, W)

        # Prepare PSF: add channel dim → (B, 1, H_m, W_m)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # Renormalize PSF and apply scale factor
        psf_p = mask / (mask.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        psf_p = psf_p * self.psf_scale

        # Compute OTF and its conjugate/magnitude-squared
        otf = psf_to_otf(psf_p, fft_shape)         # (B, 1, fft_H, fft_W//2+1)
        otf_conj = torch.conj(otf)
        otf_abs_sq = torch.abs(otf) ** 2            # |H|^2

        # Finite-difference OTFs for TV regularizer
        dx_otf, dy_otf = compute_d_otf(fft_shape, device)
        dx_abs_sq = torch.abs(dx_otf) ** 2
        dy_abs_sq = torch.abs(dy_otf) ** 2

        # Precompute C^T y (zero-pad measurement to fft_shape)
        y_padded = pad_to_fft(lensless, fft_shape)

        # CTC mask: 1 inside crop region, 0 outside
        ones_y = torch.ones(B, C, H, W, device=device)
        ctc = pad_to_fft(ones_y, fft_shape)

        # Precompute x-update denominator (constant across iterations)
        denom = mu1 * otf_abs_sq + mu2 * (dx_abs_sq + dy_abs_sq) + mu3 + 1e-8

        # Initialize ADMM variables (all zeros in padded space)
        x = torch.zeros(B, C, fft_shape[0], fft_shape[1], device=device)
        v = torch.zeros_like(x)
        ux_var = torch.zeros_like(x)
        uy_var = torch.zeros_like(x)
        w = torch.zeros_like(x)
        alpha1 = torch.zeros_like(x)
        alpha2x = torch.zeros_like(x)
        alpha2y = torch.zeros_like(x)
        alpha3 = torch.zeros_like(x)

        for _ in range(self.n_iter):
            # x-update: solve in frequency domain
            rhs_tv = finite_diff_adjoint(ux_var - alpha2x / mu2, uy_var - alpha2y / mu2)
            rhs_spatial = (
                mu1 * torch.fft.irfft2(otf_conj * torch.fft.rfft2(v - alpha1 / mu1), s=fft_shape)
                + mu2 * rhs_tv
                + mu3 * (w - alpha3 / mu3)
            )
            X = torch.fft.rfft2(rhs_spatial) / denom
            x = torch.fft.irfft2(X, s=fft_shape)

            # v-update: v = (alpha1 + mu1*Hx + C^T y) / (CTC + mu1)
            Hx_padded = torch.fft.irfft2(otf * torch.fft.rfft2(x), s=fft_shape)
            v = (alpha1 + mu1 * Hx_padded + y_padded) / (ctc + mu1)

            # u-update: anisotropic TV proximal step
            # Use tau directly as threshold (matching original Le-ADMM reference code),
            # NOT tau/mu2. With tau=2e-4 and [0,1] images, tau/mu2=2.0 would zero out
            # all finite differences (max ~1.0), completely disabling TV regularization.
            dx, dy = finite_diff(x)
            ux_var = soft_threshold(dx + alpha2x / mu2, tau)
            uy_var = soft_threshold(dy + alpha2y / mu2, tau)

            # w-update: non-negativity projection
            w = torch.clamp(x + alpha3 / mu3, min=0.0)

            # Dual updates
            alpha1 = alpha1 + mu1 * (Hx_padded - v)
            alpha2x = alpha2x + mu2 * (dx - ux_var)
            alpha2y = alpha2y + mu2 * (dy - uy_var)
            alpha3 = alpha3 + mu3 * (x - w)

        # Crop back to original image size
        reconstruction = crop_from_fft(x, (H, W))

        # Normalize per sample: clamp negatives then divide by max.
        reconstruction = torch.clamp(reconstruction, min=0.0)
        recon_flat = reconstruction.flatten(1)
        recon_max = recon_flat.max(dim=1).values.clamp(min=1e-8)
        recon_max = recon_max[:, None, None, None]
        reconstruction = reconstruction / recon_max

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
