from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Gradient reversal primitive
# ---------------------------------------------------------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output.neg() * ctx.lambd, None


def gradient_reverse(x: Tensor, lambd: float = 1.0) -> Tensor:
    """Flip the gradient flowing back through ``x`` (scaled by ``lambd``)."""
    return _GradReverse.apply(x, lambd)


# ---------------------------------------------------------------------------
# Generic factor adversary (MLP critic with GRL input)
# ---------------------------------------------------------------------------
class FactorAdversary(nn.Module):
    """Predict a pooled target from a pooled latent via GRL + 3-layer MLP.

    Forward returns ``MSE(critic(GRL(z)), pool(target))``. Because the
    gradient back to ``z`` is reversed, minimizing this loss pushes ``z``
    AWAY from carrying target-predictive information.

    Inputs:
        z       : ``[B, D]`` or ``[B, D, T]`` — time-pooled internally
        target  : ``[B, C]`` or ``[B, C, T]`` — same pool rule
    """

    def __init__(
        self,
        latent_dim: int,
        target_dim: int,
        hidden: int = 256,
        lambd: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.target_dim = int(target_dim)
        self.lambd = float(lambd)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.target_dim),
        )

    @staticmethod
    def _pool(x: Tensor) -> Tensor:
        if x.ndim == 3:
            return x.mean(dim=-1)
        if x.ndim == 2:
            return x
        raise ValueError(
            f"FactorAdversary input must be [B,D] or [B,D,T], got {tuple(x.shape)}"
        )

    def forward(
        self,
        z: Tensor,
        target: Tensor,
        lambd: Optional[float] = None,
    ) -> Tensor:
        z = self._pool(z)
        y = self._pool(target)
        z = gradient_reverse(z, self.lambd if lambd is None else float(lambd))
        pred = self.net(z)
        if pred.shape[-1] != y.shape[-1]:
            raise ValueError(
                f"FactorAdversary pred dim {pred.shape[-1]} != target dim {y.shape[-1]}"
            )
        return F.mse_loss(pred, y.detach())


# ---------------------------------------------------------------------------
# Backwards-compatible alias (Variant B ckpts still reference this name)
# ---------------------------------------------------------------------------
class PitchAdversary(FactorAdversary):
    """Alias kept for B-Codec: takes ``n_pitch_bins`` kwarg → ``target_dim``."""

    def __init__(
        self,
        latent_dim: int,
        n_pitch_bins: int,
        hidden_dim: int = 256,
        lambd: float = 1.0,
    ):
        super().__init__(
            latent_dim=latent_dim,
            target_dim=n_pitch_bins,
            hidden=hidden_dim,
            lambd=lambd,
        )


__all__ = ["FactorAdversary", "PitchAdversary", "gradient_reverse", "_GradReverse"]
