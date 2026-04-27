from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Equivariance loss for pitch
# ---------------------------------------------------------------------------
class PESTOPowerSeriesLoss(nn.Module):

    def __init__(
        self,
        n_pitch_bins: int = 128,
        bins_per_semitone: int = 1,    # 1 if n_pitch_bins is already a semitone grid
        tau: float = 1.0,
    ):
        super().__init__()
        self.n_pitch_bins = int(n_pitch_bins)
        self.bins_per_semitone = int(bins_per_semitone)
        self.tau = float(tau)

        # Per-bin geometric weight: weights[i] = q^i with q = 2^(1/(12 * bins_per_semitone))
        q_bin = 2.0 ** (1.0 / (12.0 * self.bins_per_semitone))
        powers = torch.arange(0, self.n_pitch_bins, dtype=torch.float32)
        self.register_buffer("weights", q_bin ** powers, persistent=False)
        self.q_semitone = 2.0 ** (1.0 / 12.0)
        self.huber = nn.SmoothL1Loss(beta=tau, reduction="mean")

    def _project(self, z_p: Tensor) -> Tensor:

        # weights: [n_pitch_bins]
        w = self.weights[None, None, :, None]              # [1,1,n_bins,1]
        # [B, K, n_bins, T] * [1,1,n_bins,1] → sum over n_bins → [B, K, T]
        per_kt = (z_p * w).sum(dim=2)
        # Collapse (K, T) into a single positive scalar per sample.
        return per_kt.mean(dim=(1, 2))                     # [B]

    def forward(
        self,
        z_p_anchor: Tensor,
        z_p_shifted: Tensor,
        n_semitones: Tensor,
    ) -> Tensor:

        z_a = self._project(z_p_anchor)                    # [B]
        z_s = self._project(z_p_shifted)                   # [B]
        n = n_semitones.float()
        ratio_target = self.q_semitone ** n                # [B]

        # z is positive since softmax weights are positive
        eps = 1e-6
        r_sa = z_s / (z_a + eps)
        r_as = z_a / (z_s + eps)
        target_inv = 1.0 / ratio_target.clamp_min(eps)

        # Huber on the residuals
        loss_sa = self.huber(r_sa, ratio_target)
        loss_as = self.huber(r_as, target_inv)
        return 0.5 * (loss_sa + loss_as)


# ---------------------------------------------------------------------------
# Quinton ratio equivariance loss for rhythm (time stretch)
# ---------------------------------------------------------------------------
class QuintonRatioLoss(nn.Module):

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=tau, reduction="mean")

    def forward(
        self,
        z_r_scalar_i: Tensor,      # [B]
        z_r_scalar_j: Tensor,      # [B]
        alpha_i: Tensor,           # [B], time-stretch factor for view i
        alpha_j: Tensor,           # [B], time-stretch factor for view j
    ) -> Tensor:
        eps = 1e-6

        z_i = F.softplus(z_r_scalar_i) + eps
        z_j = F.softplus(z_r_scalar_j) + eps
        ratio_z = z_i / z_j
        ratio_a = alpha_i.float() / alpha_j.float().clamp_min(eps)
        return self.huber(ratio_z, ratio_a)


# ---------------------------------------------------------------------------
# Timbre additive equivariance loss (for EQ tilt / brightness in dB)
# ---------------------------------------------------------------------------
class TimbreAdditiveLoss(nn.Module):

    def __init__(self, tau: float = 1.0, gamma_init: float = 1.0):
        super().__init__()
        # γ is a learnable scalar, initialized to 1.0 (dB ≈ brightness unit)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.huber = nn.SmoothL1Loss(beta=tau, reduction="mean")

    def forward(
        self,
        b_anchor: Tensor,          # [B], brightness of z_t_anchor
        b_shifted: Tensor,         # [B], brightness of z_t after EQ tilt
        delta_db: Tensor,          # [B], applied EQ tilt in dB
    ) -> Tensor:
        target = self.gamma * delta_db.float()
        residual = (b_shifted - b_anchor) - target
        return self.huber(residual, torch.zeros_like(residual))


# ---------------------------------------------------------------------------
# Convenience: unified scalar-equivariance loss container
# ---------------------------------------------------------------------------
class ScalarEquivLosses(nn.Module):

    def __init__(
        self,
        n_pitch_bins: int = 128,
        bins_per_semitone: int = 1,
        tau_p: float = 1.0,
        tau_r: float = 1.0,
        tau_t: float = 1.0,
        gamma_t_init: float = 1.0,
    ):
        super().__init__()
        self.pitch = PESTOPowerSeriesLoss(n_pitch_bins=n_pitch_bins, bins_per_semitone=bins_per_semitone, tau=tau_p)
        self.rhythm = QuintonRatioLoss(tau=tau_r)
        self.timbre = TimbreAdditiveLoss(tau=tau_t, gamma_init=gamma_t_init)
