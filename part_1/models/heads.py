from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Distribution head
# ---------------------------------------------------------------------------
class DistributionHead(nn.Module):

    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hidden_dim = int(hidden_dim or 0)
        if self.hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Conv1d(in_dim, self.hidden_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(self.hidden_dim, latent_dim * 2, kernel_size=1),
            )
        else:
            self.proj = nn.Conv1d(in_dim, latent_dim * 2, kernel_size=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"DistributionHead expects [B, D, T], got {tuple(x.shape)}")
        h = self.proj(x)
        mu, log_var = h.chunk(2, dim=1)
        log_var = log_var.clamp(min=-30.0, max=20.0)
        return mu, log_var


def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# Gradient reversal + adversary classes moved to losses/grl_adv.py (2026-04-16).
# Re-exported here so existing imports (``from models.heads import PitchAdversary``)
# keep working during transition; new code should import from ``losses.grl_adv``.
from losses.grl_adv import (  # noqa: E402, F401
    FactorAdversary,
    PitchAdversary,
    _GradReverse,
    gradient_reverse,
)


# ===========================================================================
# B-Codec v2: Structured heads (Plan 1 & Plan 2)
# ===========================================================================

class ToeplitzLinear(nn.Conv1d):

    def __init__(self, in_features: int, out_features: int, init: str = "identity"):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features + out_features - 1,
            padding=out_features - 1,
            bias=False,
        )
        self.in_features = in_features
        self.out_features = out_features

        if init == "identity":
            with torch.no_grad():
                kernel = self.weight            # [1, 1, in + out - 1]
                kernel.zero_()
                kernel[0, 0, out_features - 1] = 1.0
                kernel.add_(torch.randn_like(kernel) * 0.01)
        elif init == "random":
            pass  
        else:
            raise ValueError(f"ToeplitzLinear init must be 'identity' or 'random', got {init!r}")

    def forward(self, input: Tensor) -> Tensor:  # type: ignore[override]

        shape = input.shape
        if shape[-1] != self.in_features:
            raise ValueError(
                f"ToeplitzLinear: last dim must be {self.in_features}, got {shape[-1]}"
            )
        x = input.reshape(-1, 1, shape[-1])                # [N, 1, in]
        y = super().forward(x)                             # [N, 1, out]
        return y.reshape(*shape[:-1], self.out_features)   # [..., out]


# ---------------------------------------------------------------------------
# AuxFiLMInject — inject aux features via FiLM with non-zero gamma init
# ---------------------------------------------------------------------------
class AuxFiLMInject(nn.Module):

    def __init__(self, aux_dim: int, feat_dim: int, gamma_init: float = 0.1, kernel: int = 3):
        super().__init__()
        self.aux_dim = int(aux_dim)
        self.feat_dim = int(feat_dim)
        self.proj = nn.Conv1d(aux_dim, feat_dim * 2, kernel_size=kernel, padding=kernel // 2)

        fan_in = max(aux_dim * kernel, 1)
        gamma_std = float(gamma_init) / math.sqrt(fan_in)
        with torch.no_grad():
            nn.init.normal_(self.proj.weight[:feat_dim], mean=0.0, std=gamma_std)
            nn.init.zeros_(self.proj.weight[feat_dim:])
            nn.init.zeros_(self.proj.bias)

    def forward(self, trunk: Tensor, aux: Tensor) -> Tensor:
        if trunk.shape[-1] != aux.shape[-1]:
            aux = F.interpolate(aux, size=trunk.shape[-1], mode="linear", align_corners=False)
        gb = self.proj(aux)                                # [B, 2*feat_dim, T]
        gamma, beta = gb.chunk(2, dim=1)
        return (1.0 + gamma) * trunk + beta


# ---------------------------------------------------------------------------
# PitchHeadToeplitz — produces [B, K, n_pitch_bins, T] 
# ---------------------------------------------------------------------------
class PitchHeadToeplitz(nn.Module):

    def __init__(
        self,
        feat_dim: int,
        K: int = 4,
        n_pitch_bins: int = 128,
        hidden: int = 256,
        kernel: int = 3,
        use_toeplitz: bool = True,
        conv_out_gain: float = 5.0,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.K = int(K)
        self.n_pitch_bins = int(n_pitch_bins)
        self.hidden = int(hidden)
        self.use_toeplitz = bool(use_toeplitz)

        self.conv_temporal = nn.Conv1d(feat_dim, hidden, kernel, padding=kernel // 2)
        self.conv_out = nn.Conv1d(hidden, K * n_pitch_bins, 1)
        with torch.no_grad():
            self.conv_out.weight.mul_(float(conv_out_gain))

        if self.use_toeplitz:
            self.toeplitz = ToeplitzLinear(n_pitch_bins, n_pitch_bins, init="identity")
        else:
            self.toeplitz = nn.Linear(n_pitch_bins, n_pitch_bins, bias=False)

    def forward(self, h: Tensor) -> Tensor:
        """h: [B, feat_dim, T] → z_p: [B, K, n_pitch_bins, T]."""
        B, _, T = h.shape
        x = F.silu(self.conv_temporal(h))
        x = self.conv_out(x)                         # [B, K*n_bins, T]
        x = x.view(B, self.K, self.n_pitch_bins, T)  # [B, K, n_bins, T]

        xr = x.permute(0, 1, 3, 2).contiguous()      # [B, K, T, n_bins]
        xr_flat = xr.view(-1, self.n_pitch_bins)     # [B*K*T, n_bins]
        xr_flat = self.toeplitz(xr_flat)             # [B*K*T, n_bins]
        # First .contiguous() is REQUIRED before .view() on a permuted tensor.
        # Second permute is left non-contiguous: softmax handles any stride,
        # and downstream consumers call .reshape() which re-contiguifies on
        # demand. Saves ~3.5 MB copy per forward (bs=2,K=4,n_bins=128,T=861).
        xr = xr_flat.view(B, self.K, T, self.n_pitch_bins)
        x = xr.permute(0, 1, 3, 2)                   # [B, K, n_bins, T] non-contig

        z_p = F.softmax(x, dim=2)
        return z_p


# ---------------------------------------------------------------------------
# RhythmHeadConv 
# ---------------------------------------------------------------------------
class RhythmHeadConv(nn.Module):

    def __init__(self, feat_dim: int, d_r: int = 64, hidden: int = 128, kernel: int = 5):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.d_r = int(d_r)
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, kernel, padding=kernel // 2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
            nn.SiLU(),
            nn.Conv1d(hidden, d_r, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)  # [B, d_r, T]


# ---------------------------------------------------------------------------
# TimbreHeadGlobal 
# ---------------------------------------------------------------------------
class TimbreHeadGlobal(nn.Module):

    def __init__(self, feat_dim: int, d_t: int = 128, hidden: int = 256, kernel: int = 3):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.d_t = int(d_t)
        self.hidden = int(hidden)

        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, kernel, padding=kernel // 2),
            nn.SiLU(),
            nn.Conv1d(hidden, d_t, 1),
        )
        self.refine = nn.Linear(d_t, d_t)
        self.norm = nn.LayerNorm(d_t)

    def forward(self, h: Tensor) -> Tensor:
        x = self.conv(h)                     # [B, d_t, T]
        x = x.mean(dim=-1)                   # [B, d_t]  time-pool
        x = self.refine(x)                   # [B, d_t]
        return self.norm(x)                  # [B, d_t] unit-var


# ---------------------------------------------------------------------------
# TimbreFiLMGen — per-layer (γ, β)
# ---------------------------------------------------------------------------
class TimbreFiLMGen(nn.Module):

    def __init__(
        self,
        d_t: int,
        hidden_dims: Sequence[int],
        gamma_init: float = 0.1,
        gamma_clamp: float = 0.5,
    ):
        super().__init__()
        self.d_t = int(d_t)
        self.hidden_dims = list(hidden_dims)
        self.gamma_clamp = float(gamma_clamp)
        self.layers = nn.ModuleList(
            [nn.Linear(d_t, 2 * h) for h in hidden_dims]
        )

        fan_in = max(d_t, 1)
        gamma_std = float(gamma_init) / math.sqrt(fan_in)
        with torch.no_grad():
            for ly in self.layers:
                H = ly.weight.shape[0] // 2
                nn.init.normal_(ly.weight[:H], mean=0.0, std=gamma_std)  # gamma rows
                nn.init.zeros_(ly.weight[H:])                            # beta rows
                nn.init.zeros_(ly.bias)

    def forward(self, z_t: Tensor) -> List[Tuple[Tensor, Tensor]]:
        outs: List[Tuple[Tensor, Tensor]] = []
        for ly in self.layers:
            gb = ly(z_t)                                   # [B, 2h]
            gamma_raw, beta = gb.chunk(2, dim=-1)
            gamma = torch.tanh(gamma_raw) * self.gamma_clamp
            outs.append((gamma.unsqueeze(-1), beta.unsqueeze(-1)))  # broadcast over T
        return outs


# ---------------------------------------------------------------------------
# PitchEmbed — compress [B, K, n_bins, T] softmax distribution → [B, d_p, T]
# ---------------------------------------------------------------------------
class PitchEmbed(nn.Module):

    def __init__(self, K: int, n_pitch_bins: int, d_out: int = 128, hidden: int = 256):
        super().__init__()
        self.K = K
        self.n_pitch_bins = n_pitch_bins
        self.d_out = d_out
        in_ch = K * n_pitch_bins
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1),
            nn.SiLU(),
            nn.Conv1d(hidden, d_out, 1),
        )

    def forward(self, z_p: Tensor) -> Tensor:
        B, K, NB, T = z_p.shape
        assert K == self.K and NB == self.n_pitch_bins, (
            f"PitchEmbed shape mismatch: expected K={self.K}, NB={self.n_pitch_bins}, "
            f"got K={K}, NB={NB}"
        )
        x = z_p.reshape(B, K * NB, T)
        return self.net(x)


# ---------------------------------------------------------------------------
# RhythmScalarProjector / TimbreBrightnessProjector 
# ---------------------------------------------------------------------------
class RhythmScalarProjector(nn.Module):

    def __init__(self, d_r: int):
        super().__init__()
        self.proj = nn.Linear(d_r, 1, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(d_r))
            self.proj.weight.abs_()

    def forward(self, z_r: Tensor) -> Tensor:
        # [B, d_r, T] → mean over T → [B, d_r] → Linear → [B]
        pooled = z_r.mean(dim=-1)
        return self.proj(pooled).squeeze(-1)


class TimbreBrightnessProjector(nn.Module):

    def __init__(self, d_t: int):
        super().__init__()
        self.proj = nn.Linear(d_t, 1, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(d_t))

    def forward(self, z_t: Tensor) -> Tensor:
        return self.proj(z_t).squeeze(-1)
