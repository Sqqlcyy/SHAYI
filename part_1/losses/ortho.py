from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _canonicalize(z: Tensor) -> Tensor:
    if z.ndim == 4:
        B, K, D, T = z.shape
        return z.reshape(B, K * D, T)
    if z.ndim in (2, 3):
        return z
    raise ValueError(
        f"latent must be [B,D], [B,D,T], or [B,K,D,T], got {tuple(z.shape)}"
    )


def _pool_time(z: Tensor) -> Tensor:
    z = _canonicalize(z)
    if z.ndim == 3:
        return z.mean(dim=-1)
    if z.ndim == 2:
        return z
    raise ValueError(f"latent must be [B, D] or [B, D, T], got {tuple(z.shape)}")


def _center(z: Tensor) -> Tensor:
    return z - z.mean(dim=0, keepdim=True)


def _align_time(z: Tensor, target_T: int) -> Tensor:
    z = _canonicalize(z)
    if z.ndim == 2:
        return z.unsqueeze(-1).expand(-1, -1, target_T)
    if z.ndim == 3:
        if z.shape[-1] == target_T:
            return z
        return F.interpolate(z, size=target_T, mode="linear", align_corners=False)
    raise ValueError(f"latent must be [B, D] or [B, D, T], got {tuple(z.shape)}")


def _flatten_for_cross_cov(z_a: Tensor, z_b: Tensor) -> Tuple[Tensor, Tensor]:

    z_a = _canonicalize(z_a)
    z_b = _canonicalize(z_b)
    if z_a.ndim == 2 and z_b.ndim == 2:
        return z_a, z_b
    if z_a.ndim == 3 and z_b.ndim == 3:
        target_T = max(z_a.shape[-1], z_b.shape[-1])
        a_t = _align_time(z_a, target_T)
        b_t = _align_time(z_b, target_T)
        a = a_t.permute(0, 2, 1).reshape(-1, a_t.shape[1])
        b = b_t.permute(0, 2, 1).reshape(-1, b_t.shape[1])
        return a, b
    a2 = _pool_time(z_a)
    b2 = _pool_time(z_b)
    return a2, b2


# ---------------------------------------------------------------------------
# Cross covariance Frobenius
# ---------------------------------------------------------------------------
def cross_cov_loss(z_a: Tensor, z_b: Tensor) -> Tensor:
    """Frobenius norm of the sample cross-covariance matrix, normalized by d_a*d_b.

    z_a: [B, D_a] or [B, D_a, T]
    z_b: [B, D_b] or [B, D_b, T]
    """
    a, b = _flatten_for_cross_cov(z_a, z_b)
    a = _center(a)
    b = _center(b)
    N = a.shape[0]
    cov = (a.t() @ b) / max(N - 1, 1)  # [D_a, D_b]
    norm_sq = (cov ** 2).sum()
    return norm_sq / (a.shape[1] * b.shape[1])


# ---------------------------------------------------------------------------
# HSIC with Gaussian kernels
# ---------------------------------------------------------------------------
def _gaussian_gram(x: Tensor, sigma: float = None) -> Tensor:
    """Gram matrix with RBF kernel. Sigma defaults to median-distance heuristic."""
    B = x.shape[0]
    # pairwise sq distances
    sq = torch.cdist(x, x, p=2) ** 2
    if sigma is None:
        with torch.no_grad():
            med = sq[sq > 0].median() if (sq > 0).any() else torch.tensor(1.0, device=x.device)
            sigma = torch.sqrt(med / 2.0 + 1e-8)
    K = torch.exp(-sq / (2.0 * sigma ** 2 + 1e-8))
    return K


def hsic_loss(z_a: Tensor, z_b: Tensor) -> Tensor:
    
    a = _pool_time(z_a)
    b = _pool_time(z_b)
    B = a.shape[0]
    Ka = _gaussian_gram(a)
    Kb = _gaussian_gram(b)
    H = torch.eye(B, device=a.device) - (1.0 / B) * torch.ones(B, B, device=a.device)
    HKaH = H @ Ka @ H
    HKbH = H @ Kb @ H
    hsic = (HKaH * HKbH).sum() / ((B - 1) ** 2 + 1e-8)
    return hsic


# ---------------------------------------------------------------------------
# Composite ortho loss over the 3 pair combinations
# ---------------------------------------------------------------------------
class OrthoLoss(nn.Module):
    """Sum of the 3 pair losses for (z_p, z_r), (z_p, z_t), (z_r, z_t)."""

    def __init__(self, form: str = "cross_cov"):
        super().__init__()
        if form not in ("cross_cov", "hsic"):
            raise ValueError(f"Unknown ortho form: {form}")
        self.form = form
        self._fn = cross_cov_loss if form == "cross_cov" else hsic_loss

    def forward(self, z_p: Tensor, z_r: Tensor, z_t: Tensor) -> Dict[str, Tensor]:
        l_pr = self._fn(z_p, z_r)
        l_pt = self._fn(z_p, z_t)
        l_rt = self._fn(z_r, z_t)
        total = l_pr + l_pt + l_rt
        return {
            "ortho/total": total,
            "ortho/pr": l_pr.detach(),
            "ortho/pt": l_pt.detach(),
            "ortho/rt": l_rt.detach(),
        }
