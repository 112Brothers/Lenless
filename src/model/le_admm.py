"""
Unrolled Learned ADMM (Le-ADMM) for lensless reconstruction.
mu and tau are learnable per iteration.

Reference: "Learned reconstructions for practical mask-based lensless imaging"
           https://arxiv.org/abs/1908.11502

Hyperparameters per task spec: mu_i = 1e-4, tau_i = 2e-4 (initial values).
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


class LeADMM(nn.Module):
    """
    Unrolled Learned ADMM (Le-ADMM) for lensless reconstruction.
    mu and tau are learnable per iteration (stored as log for positivity).

    ADMM solves: min_x 0.5*||Hx - y||^2 + tau*TV(x)
    via variable splitting with z = Dx, u = dual variable.

    x-update: (H^T H + mu*(Dx^T Dx + Dy^T Dy)) x = H^T y + mu*D^T(z - u)
    z-update: z = soft_threshold(Dx + u, tau/mu)
    u-update: u = u + Dx - z
    """

    def __init__(self, n_iter=20, init_mu=1e-4, init_tau=2e-4):
        super().__init__()
        self.n_iter = n_iter

        # Learnable parameters per iteration, stored as log for positivity.
        # Task spec: mu_i = 1e-4, tau_i = 2e-4 (initial values).
        self.log_mu = nn.ParameterList([
            nn.Parameter(torch.tensor(float(init_mu)).log())
            for _ in range(n_iter)
        ])
        self.log_tau = nn.ParameterList([
            nn.Parameter(torch.tensor(float(init_tau)).log())
            for _ in range(n_iter)
        ])

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

        # Compute padded FFT shape (~2H x 2W, FFT-friendly)
        fft_shape = compute_fft_shape(H, W)

        # Prepare PSF: add channel dim → (B, 1, H_m, W_m)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # Compute OTF and its conjugate/magnitude-squared
        otf = psf_to_otf(mask, fft_shape)          # (B, 1, fft_H, fft_W//2+1)
        otf_conj = torch.conj(otf)
        otf_abs_sq = torch.abs(otf) ** 2            # |H|^2

        # Finite-difference OTFs for TV regularizer
        dx_otf, dy_otf = compute_d_otf(fft_shape, device)
        dx_abs_sq = torch.abs(dx_otf) ** 2
        dy_abs_sq = torch.abs(dy_otf) ** 2

        # Precompute H^T y in frequency domain
        y_padded = pad_to_fft(lensless, fft_shape)
        Y = torch.fft.rfft2(y_padded)
        ATy = otf_conj * Y                          # (B, C, fft_H, fft_W//2+1)

        # Initialize ADMM variables (all zeros in padded space)
        zx = torch.zeros(B, C, fft_shape[0], fft_shape[1], device=device)
        zy = torch.zeros_like(zx)
        ux = torch.zeros_like(zx)
        uy = torch.zeros_like(zx)

        for i in range(self.n_iter):
            mu = torch.exp(self.log_mu[i])
            tau = torch.exp(self.log_tau[i])

            # x-update: solve in frequency domain
            # (H^T H + mu*(Dx^T Dx + Dy^T Dy)) X = H^T y + mu * D^T(z - u)
            denom = otf_abs_sq + mu * (dx_abs_sq + dy_abs_sq) + 1e-8
            rhs_spatial = finite_diff_adjoint(zx - ux, zy - uy)
            RHS = ATy + mu * torch.fft.rfft2(rhs_spatial)
            X = RHS / denom
            x = torch.fft.irfft2(X, s=fft_shape)

            # Compute finite differences of x
            dx, dy = finite_diff(x)

            # z-update: anisotropic TV proximal step
            # prox_{(tau/mu)*||.||_1}(v) = soft_threshold(v, tau/mu)
            zx = soft_threshold(dx + ux, tau / mu)
            zy = soft_threshold(dy + uy, tau / mu)

            # u-update (scaled dual variable)
            ux = ux + dx - zx
            uy = uy + dy - zy

        # Extra x-update after the loop so that log_tau[-1] receives gradient.
        # (log_tau[-1] only affects z[-1], which feeds into the final u-update,
        #  which feeds into this extra x-update.)
        mu_last = torch.exp(self.log_mu[-1])
        denom_last = otf_abs_sq + mu_last * (dx_abs_sq + dy_abs_sq) + 1e-8
        rhs_spatial = finite_diff_adjoint(zx - ux, zy - uy)
        RHS = ATy + mu_last * torch.fft.rfft2(rhs_spatial)
        X = RHS / denom_last
        x = torch.fft.irfft2(X, s=fft_shape)

        # Crop back to original image size
        reconstruction = crop_from_fft(x, (H, W))

        # Normalize per sample: clamp negatives then divide by max.
        # This matches the Le-ADMM paper's normalize_image() function.
        reconstruction = torch.clamp(reconstruction, min=0.0)
        recon_flat = reconstruction.flatten(1)                          # (B, C*H*W)
        recon_max = recon_flat.max(dim=1).values.clamp(min=1e-8)       # (B,)
        recon_max = recon_max[:, None, None, None]                      # (B, 1, 1, 1)
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
