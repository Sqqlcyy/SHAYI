from __future__ import annotations

import torch
from torch import Tensor


def variance_of_energy_loss(z: Tensor, k: int) -> Tensor:
    """z: [B, D] or [B, D, T]; split D into k contiguous groups."""
    if z.ndim == 3:
        # average energy over time first
        z = z.pow(2).mean(dim=-1)  # [B, D]
    elif z.ndim == 2:
        z = z.pow(2)
    else:
        raise ValueError(f"z must be [B, D] or [B, D, T], got {tuple(z.shape)}")

    B, D = z.shape
    if D % k != 0:
        raise ValueError(f"latent dim D={D} must be divisible by k={k}")
    z = z.view(B, k, D // k).mean(dim=-1)  # [B, k] energy per group
    # coefficient of variation squared, averaged over batch
    mean = z.mean(dim=-1, keepdim=True)             # [B, 1]
    var = ((z - mean) ** 2).mean(dim=-1)            # [B]
    cv2 = var / (mean.squeeze(-1) ** 2 + 1e-8)      # [B]
    return cv2.mean()
