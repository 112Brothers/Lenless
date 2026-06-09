"""
Unrolled Learned ADMM (Le-ADMM) for lensless reconstruction.
mu and tau are learnable per iteration.
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
    mu and tau are learnable per iteration.
    """

    def __init__(self, n_iter=20, init_mu=1e-4, init_tau=2e-4):
        super().__init__()
        self.n_iter = n_iter

        # Learnable parameters per iteration
        # Store as log values to ensure positivity via exp()
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
            lensless: (B, C, H, W) lensless measurement
            mask: (B, H_m, W_m) PSF/mask
        Returns:
            dict with "reconstruction": (B, C, H, W)
        """
        B, C, H, W = lensless.shape
        device = lensless.device

        fft_shape = compute_fft_shape(H, W)

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        otf = psf_to_otf(mask, fft_shape)
        otf_conj = torch.conj(otf)
        otf_abs_sq = torch.abs(otf) ** 2

        dx_otf, dy_otf = compute_d_otf(fft_shape, device)
        dx_abs_sq = torch.abs(dx_otf) ** 2
        dy_abs_sq = torch.abs(dy_otf) ** 2

        y_padded = pad_to_fft(lensless, fft_shape)
        Y = torch.fft.rfft2(y_padded)
        ATy = otf_conj * Y

        # Initialize variables
        zx = torch.zeros(B, C, fft_shape[0], fft_shape[1], device=device)
        zy = torch.zeros_like(zx)
        ux = torch.zeros_like(zx)
        uy = torch.zeros_like(zx)

        for i in range(self.n_iter):
            mu = torch.exp(self.log_mu[i])
            tau = torch.exp(self.log_tau[i])

            # x-update: denominator changes each iteration due to mu
            denom = otf_abs_sq + mu * (dx_abs_sq + dy_abs_sq) + 1e-8

            rhs_spatial = finite_diff_adjoint(
                zx - ux, zy - uy
            )
            RHS = ATy + mu * torch.fft.rfft2(rhs_spatial)
            X = RHS / denom
            x = torch.fft.irfft2(X, s=fft_shape)

            dx, dy = finite_diff(x)

            # z-update
            zx = soft_threshold(dx + ux, tau / mu)
            zy = soft_threshold(dy + uy, tau / mu)

            # u-update
            ux = ux + dx - zx
            uy = uy + dy - zy

        # Final x-update after all iterations to ensure all parameters affect output
        # This ensures tau[n_iter-1] has gradient flow
        mu = torch.exp(self.log_mu[-1])
        rhs_spatial = finite_diff_adjoint(zx - ux, zy - uy)
        RHS = ATy + mu * torch.fft.rfft2(rhs_spatial)
        X = RHS / denom
        x = torch.fft.irfft2(X, s=fft_shape)

        reconstruction = crop_from_fft(x, (H, W))
        # Use sigmoid instead of clamp to allow gradients to flow
        # clamp blocks gradients for out-of-range values
        reconstruction = torch.sigmoid(reconstruction)

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
