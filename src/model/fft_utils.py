"""
FFT-based utilities for lensless imaging forward model and ADMM.

The lensless forward model is: y = crop(h * x)
where * is convolution with PSF h, and crop extracts the center region.

Strategy: work in padded space (≥2H × 2W, rounded to FFT-friendly size).
"""

import torch
import torch.nn.functional as F


def next_fast_fft_size(n):
    """Find the next FFT-friendly size >= n (product of small primes 2,3,5)."""
    while True:
        m = n
        for p in [2, 3, 5]:
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def compute_fft_shape(H, W, factor=2):
    """Compute padded FFT shape from image dimensions."""
    fft_H = next_fast_fft_size(factor * H)
    fft_W = next_fast_fft_size(factor * W)
    return fft_H, fft_W


def pad_to_fft(x, fft_shape):
    """
    Zero-pad tensor x to fft_shape, centering the original content.

    Args:
        x: (B, C, H, W) tensor
        fft_shape: (fft_H, fft_W) target shape
    Returns:
        padded: (B, C, fft_H, fft_W) tensor
    """
    _, _, H, W = x.shape
    fft_H, fft_W = fft_shape
    pad_top = (fft_H - H) // 2
    pad_bottom = fft_H - H - pad_top
    pad_left = (fft_W - W) // 2
    pad_right = fft_W - W - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)


def crop_from_fft(x, target_shape):
    """
    Crop center region from padded tensor.

    Args:
        x: (B, C, fft_H, fft_W) tensor
        target_shape: (H, W) desired output shape
    Returns:
        cropped: (B, C, H, W) tensor
    """
    _, _, fft_H, fft_W = x.shape
    H, W = target_shape
    start_h = (fft_H - H) // 2
    start_w = (fft_W - W) // 2
    return x[:, :, start_h:start_h + H, start_w:start_w + W]


def psf_to_otf(psf, fft_shape):
    """
    Convert PSF to OTF (Optical Transfer Function).

    Args:
        psf: (B, C, H, W) or (B, 1, H, W) PSF tensor (already normalized)
        fft_shape: (fft_H, fft_W) target FFT shape
    Returns:
        otf: (B, C, fft_H, fft_W//2+1) complex tensor
    """
    # Pad PSF to fft_shape
    psf_padded = pad_to_fft(psf, fft_shape)
    # Use ifftshift to center the PSF before FFT
    # (PSF is typically centered, we need it at origin for FFT)
    psf_padded = torch.fft.ifftshift(psf_padded, dim=(-2, -1))
    otf = torch.fft.rfft2(psf_padded)
    return otf


def compute_d_otf(fft_shape, device):
    """
    Compute OTFs for finite difference operators (circular TV).

    Dx[i,j] = x[i, (j+1) % W] - x[i, j]  (horizontal)
    Dy[i,j] = x[(i+1) % H, j] - x[i, j]  (vertical)

    Args:
        fft_shape: (fft_H, fft_W)
        device: torch device
    Returns:
        dx_otf: (1, 1, fft_H, fft_W//2+1) complex tensor
        dy_otf: (1, 1, fft_H, fft_W//2+1) complex tensor
    """
    fft_H, fft_W = fft_shape

    # Horizontal difference kernel: [1, -1] at positions [0,0] and [0,1]
    dx_kernel = torch.zeros(1, 1, fft_H, fft_W, device=device)
    dx_kernel[0, 0, 0, 0] = -1.0
    dx_kernel[0, 0, 0, 1] = 1.0
    dx_otf = torch.fft.rfft2(dx_kernel)

    # Vertical difference kernel: [1; -1] at positions [0,0] and [1,0]
    dy_kernel = torch.zeros(1, 1, fft_H, fft_W, device=device)
    dy_kernel[0, 0, 0, 0] = -1.0
    dy_kernel[0, 0, 1, 0] = 1.0
    dy_otf = torch.fft.rfft2(dy_kernel)

    return dx_otf, dy_otf


def finite_diff(x):
    """
    Circular finite differences for anisotropic TV.

    Args:
        x: (B, C, H, W) tensor
    Returns:
        dx: (B, C, H, W) horizontal differences
        dy: (B, C, H, W) vertical differences
    """
    dx = torch.roll(x, shifts=-1, dims=-1) - x  # circular horizontal
    dy = torch.roll(x, shifts=-1, dims=-2) - x  # circular vertical
    return dx, dy


def finite_diff_adjoint(dx, dy):
    """
    Adjoint of circular finite differences.

    For Dx[i,j] = x[i, j+1] - x[i, j]:
      <Dx, v> = sum x[i,j]*(v[i,j-1] - v[i,j])
      => D^T_x v[i,j] = v[i, j-1] - v[i, j] = roll(v, +1) - v

    Args:
        dx: (B, C, H, W) horizontal differences
        dy: (B, C, H, W) vertical differences
    Returns:
        result: (B, C, H, W) adjoint output
    """
    # Adjoint of Dx: DT_x[i,j] = roll(dx, +1)[i,j] - dx[i,j]
    adj_dx = torch.roll(dx, shifts=1, dims=-1) - dx
    # Adjoint of Dy: DT_y[i,j] = roll(dy, +1)[i,j] - dy[i,j]
    adj_dy = torch.roll(dy, shifts=1, dims=-2) - dy
    return adj_dx + adj_dy


def soft_threshold(x, threshold):
    """
    Element-wise soft thresholding.
    sign(x) * max(|x| - threshold, 0)
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)
