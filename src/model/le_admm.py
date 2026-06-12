"""
Unrolled Learned ADMM (Le-ADMM) for lensless reconstruction.
mu1, mu2, mu3, tau are learnable per iteration.

Reference: "Learned reconstructions for practical mask-based lensless imaging"
           Monakhova et al., Opt. Express 2019. https://arxiv.org/abs/1908.11502

ADMM solves: min_x 0.5*||C*H*x - y||^2 + tau*TV(x)  s.t. x >= 0
via 3-variable splitting:
  v = C*H*x  (data fidelity, crop operator C)
  u = D*x    (TV, finite differences D)
  w = x      (non-negativity)

x-update: (mu1*H^T C^T C H + mu2*D^T D + mu3*I) x = mu1*H^T C^T(v - alpha1/mu1)
                                                     + mu2*D^T(u - alpha2/mu2)
                                                     + mu3*(w - alpha3/mu3)
u-update: u = soft_threshold(D*x + alpha2/mu2, tau/mu2)
v-update: v = (C*H*x + alpha1/mu1 + Cty/mu1) / (1 + 1/mu1)  [simplified]
w-update: w = max(x + alpha3/mu3, 0)
dual updates: alpha_i += mu_i * (primal_residual_i)

Hyperparameters per task spec: mu1=mu2=mu3=1e-4, tau=2e-4 (initial values).
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
    mu1, mu2, mu3, tau are learnable per iteration (stored as log for positivity).

    3-variable splitting ADMM:
      v = C*H*x  (data fidelity with crop C)
      u = D*x    (TV regularization)
      w = x      (non-negativity constraint)

    x-update denominator: mu1*|H|^2 + mu2*(|Dx|^2 + |Dy|^2) + mu3
    """

    def __init__(self, n_iter=20, init_mu=1e-4, init_tau=2e-4):
        super().__init__()
        self.n_iter = n_iter

        # Learnable parameters per iteration, stored as log for positivity.
        # Task spec: mu_i = 1e-4, tau_i = 2e-4 (initial values).
        self.log_mu1 = nn.ParameterList([
            nn.Parameter(torch.tensor(float(init_mu)).log())
            for _ in range(n_iter)
        ])
        self.log_mu2 = nn.ParameterList([
            nn.Parameter(torch.tensor(float(init_mu)).log())
            for _ in range(n_iter)
        ])
        self.log_mu3 = nn.ParameterList([
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

        # Precompute H^T C^T y in frequency domain
        # C^T y = zero-pad y to fft_shape (C is the crop operator)
        y_padded = pad_to_fft(lensless, fft_shape)
        Y = torch.fft.rfft2(y_padded)
        ATy = otf_conj * Y                          # H^T C^T y

        # Initialize ADMM variables (all zeros in padded space)
        x = torch.zeros(B, C, fft_shape[0], fft_shape[1], device=device)
        # v: data fidelity variable (in padded space, represents C*H*x)
        v = torch.zeros_like(x)
        # u: TV variable (x and y finite differences)
        ux_var = torch.zeros_like(x)
        uy_var = torch.zeros_like(x)
        # w: non-negativity variable
        w = torch.zeros_like(x)
        # Dual variables
        alpha1 = torch.zeros_like(x)   # dual for v = C*H*x
        alpha2x = torch.zeros_like(x)  # dual for ux = Dx*x
        alpha2y = torch.zeros_like(x)  # dual for uy = Dy*x
        alpha3 = torch.zeros_like(x)   # dual for w = x

        for i in range(self.n_iter):
            mu1 = torch.exp(self.log_mu1[i])
            mu2 = torch.exp(self.log_mu2[i])
            mu3 = torch.exp(self.log_mu3[i])
            tau = torch.exp(self.log_tau[i])

            # x-update: solve in frequency domain
            # (mu1*H^T H + mu2*(Dx^T Dx + Dy^T Dy) + mu3*I) X =
            #   mu1*H^T(v - alpha1/mu1) + mu2*D^T(u - alpha2/mu2) + mu3*(w - alpha3/mu3)
            denom = mu1 * otf_abs_sq + mu2 * (dx_abs_sq + dy_abs_sq) + mu3 + 1e-8
            rhs_tv = finite_diff_adjoint(ux_var - alpha2x / mu2, uy_var - alpha2y / mu2)
            rhs_spatial = (
                mu1 * torch.fft.irfft2(otf_conj * torch.fft.rfft2(v - alpha1 / mu1), s=fft_shape)
                + mu2 * rhs_tv
                + mu3 * (w - alpha3 / mu3)
            )
            X = torch.fft.rfft2(rhs_spatial) / denom
            x = torch.fft.irfft2(X, s=fft_shape)

            # v-update: v = (C*H*x + y/mu1 + alpha1/mu1) / (1/mu1 + 1)
            # Simplified: v = (mu1 * C*H*x + Cty + alpha1) / (mu1 + 1)
            # C*H*x in padded space = ifft(H * fft(x))
            Hx_padded = torch.fft.irfft2(otf * torch.fft.rfft2(x), s=fft_shape)
            v = (mu1 * Hx_padded + y_padded + alpha1) / (mu1 + 1.0)

            # u-update: anisotropic TV proximal step
            dx, dy = finite_diff(x)
            ux_var = soft_threshold(dx + alpha2x / mu2, tau / mu2)
            uy_var = soft_threshold(dy + alpha2y / mu2, tau / mu2)

            # w-update: non-negativity projection
            w = torch.clamp(x + alpha3 / mu3, min=0.0)

            # Dual updates
            alpha1 = alpha1 + mu1 * (Hx_padded - v)
            alpha2x = alpha2x + mu2 * (dx - ux_var)
            alpha2y = alpha2y + mu2 * (dy - uy_var)
            alpha3 = alpha3 + mu3 * (x - w)

        # Extra x-update after the loop so that log_tau[-1] receives gradient.
        mu1_last = torch.exp(self.log_mu1[-1])
        mu2_last = torch.exp(self.log_mu2[-1])
        mu3_last = torch.exp(self.log_mu3[-1])
        denom_last = mu1_last * otf_abs_sq + mu2_last * (dx_abs_sq + dy_abs_sq) + mu3_last + 1e-8
        rhs_tv = finite_diff_adjoint(ux_var - alpha2x / mu2_last, uy_var - alpha2y / mu2_last)
        rhs_spatial = (
            mu1_last * torch.fft.irfft2(otf_conj * torch.fft.rfft2(v - alpha1 / mu1_last), s=fft_shape)
            + mu2_last * rhs_tv
            + mu3_last * (w - alpha3 / mu3_last)
        )
        X = torch.fft.rfft2(rhs_spatial) / denom_last
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
