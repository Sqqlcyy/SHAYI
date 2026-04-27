# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Multi-head VAE Encoder for decoupled music representation.
Backbone (128-d) + 3 heads (Pitch 256-d, Timbre 128-d, Rhythm 128-d) with adapters,
VAE reparameterization, output concatenated latent [B, 512, T].
"""

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seanet import SEANetEncoder


# Fixed architecture constants from spec
FEAT_DIM = 128
ADAPTER_OUT_DIM = 64
PITCH_LATENT_DIM = 256
TIMBRE_LATENT_DIM = 128
RHYTHM_LATENT_DIM = 128
TOTAL_LATENT_DIM = PITCH_LATENT_DIM + TIMBRE_LATENT_DIM + RHYTHM_LATENT_DIM  # 512


class Adapter1d(nn.Module):
    """Maps auxiliary signal to 64-d at fixed time steps (e.g. 100)."""

    def __init__(self, aux_in_dim: int, out_dim: int = ADAPTER_OUT_DIM):
        super().__init__()
        self.proj = nn.Conv1d(aux_in_dim, out_dim, kernel_size=1)

    def forward(self, aux: torch.Tensor, target_length: int) -> torch.Tensor:
        # aux: [B, C, T'] -> [B, out_dim, target_length]
        if aux.dim() == 2:
            aux = aux.unsqueeze(1)  # [B, T'] -> [B, 1, T']
        if aux.shape[-1] != target_length:
            aux = F.interpolate(aux, size=target_length, mode='linear', align_corners=False)
        return self.proj(aux)


class DistributionHead(nn.Module):
    """Single head: feat(128) + cond(64) -> mu and log_var."""

    def __init__(self, feat_dim: int, cond_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Conv1d(feat_dim + cond_dim, latent_dim * 2, kernel_size=1)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # feat [B, 128, T], cond [B, 64, T]
        x = torch.cat([feat, cond], dim=1)  # [B, 192, T]
        out = self.conv(x)  # [B, latent_dim*2, T]
        mu, log_var = out.chunk(2, dim=1)
        return mu, log_var


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Reparameterization: z = mu + sigma * eps, eps ~ N(0,1)."""
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class MultiheadVAEEncoder(nn.Module):
    """
    Multi-head VAE Encoder for decoupled music generation.

    - Shared backbone: audio [B, 1, 32000] -> feat [B, 128, 100].
    - Three heads with adapters: feat + adapted aux (64-d) -> (mu, log_var) per head.
    - VAE sampling -> z_pitch [B,256,T], z_timbre [B,128,T], z_rhythm [B,128,T].
    - Output: concat -> [B, 512, 100].

    Auxiliary inputs (f0, timbre cluster, onsets) are optional; use zeros when None.
    """

    def __init__(
        self,
        backbone: nn.Module,
        aux_pitch_dim: int = 1,
        aux_timbre_dim: int = 1,
        aux_rhythm_dim: int = 1,
    ):
        super().__init__()
        self.backbone = backbone
        self.hop_length = getattr(backbone, 'hop_length', 320)
        self.dimension = TOTAL_LATENT_DIM  # 512 for decoder compatibility

        self.adapter_pitch = Adapter1d(aux_pitch_dim, ADAPTER_OUT_DIM)
        self.adapter_timbre = Adapter1d(aux_timbre_dim, ADAPTER_OUT_DIM)
        self.adapter_rhythm = Adapter1d(aux_rhythm_dim, ADAPTER_OUT_DIM)

        self.head_pitch = DistributionHead(FEAT_DIM, ADAPTER_OUT_DIM, PITCH_LATENT_DIM)
        self.head_timbre = DistributionHead(FEAT_DIM, ADAPTER_OUT_DIM, TIMBRE_LATENT_DIM)
        self.head_rhythm = DistributionHead(FEAT_DIM, ADAPTER_OUT_DIM, RHYTHM_LATENT_DIM)

    def _get_cond(
        self,
        aux: tp.Optional[torch.Tensor],
        target_length: int,
        batch_size: int,
        device: torch.device,
        adapter: nn.Module,
    ) -> torch.Tensor:
        if aux is not None:
            return adapter(aux, target_length)
        return torch.zeros(batch_size, ADAPTER_OUT_DIM, target_length, device=device, dtype=torch.float32)

    def forward(
        self,
        x: torch.Tensor,
        aux_pitch: tp.Optional[torch.Tensor] = None,
        aux_timbre: tp.Optional[torch.Tensor] = None,
        aux_rhythm: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T_audio] e.g. [B, 1, 32000]
            aux_pitch, aux_timbre, aux_rhythm: optional [B, C, T'] or [B, T']
        Returns:
            z: [B, 512, T] latent (T = 100 for 1s at 32kHz with hop 320)
        """
        feat = self.backbone(x)  # [B, 128, T]
        B, _, T = feat.shape
        device = feat.device

        cond_pitch = self._get_cond(aux_pitch, T, B, device, self.adapter_pitch)
        cond_timbre = self._get_cond(aux_timbre, T, B, device, self.adapter_timbre)
        cond_rhythm = self._get_cond(aux_rhythm, T, B, device, self.adapter_rhythm)

        mu_p, log_var_p = self.head_pitch(feat, cond_pitch)
        mu_t, log_var_t = self.head_timbre(feat, cond_timbre)
        mu_r, log_var_r = self.head_rhythm(feat, cond_rhythm)

        z_pitch = reparameterize(mu_p, log_var_p)   # [B, 256, T]
        z_timbre = reparameterize(mu_t, log_var_t)  # [B, 128, T]
        z_rhythm = reparameterize(mu_r, log_var_r)  # [B, 128, T]

        z = torch.cat([z_pitch, z_timbre, z_rhythm], dim=1)  # [B, 512, T]
        return z


def build_multihead_vae_encoder(
    sample_rate: int = 32000,
    channels: int = 1,
    ratios: tp.List[int] = [8, 5, 4, 2],
    n_filters: int = 32,
    n_residual_layers: int = 1,
    causal: bool = False,
    aux_pitch_dim: int = 1,
    aux_timbre_dim: int = 1,
    aux_rhythm_dim: int = 1,
    **backbone_kw,
) -> MultiheadVAEEncoder:
    """Build MultiheadVAEEncoder with SEANet backbone (128-d, 320x downsample)."""
    backbone = SEANetEncoder(
        channels=channels,
        dimension=FEAT_DIM,
        n_filters=n_filters,
        n_residual_layers=n_residual_layers,
        ratios=ratios,
        causal=causal,
        **(backbone_kw or {}),
    )
    return MultiheadVAEEncoder(
        backbone=backbone,
        aux_pitch_dim=aux_pitch_dim,
        aux_timbre_dim=aux_timbre_dim,
        aux_rhythm_dim=aux_rhythm_dim,
    )
