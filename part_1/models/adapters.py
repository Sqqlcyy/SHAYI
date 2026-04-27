from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _prepare_aux(aux: Tensor, target_length: int) -> Tensor:

    if aux.ndim == 2:
        if aux.shape[1] <= 32:
            return aux.unsqueeze(-1).expand(-1, -1, target_length).contiguous()
        aux = aux.unsqueeze(1)
    if aux.ndim != 3:
        raise ValueError(f"aux must be 2D or 3D tensor, got shape={tuple(aux.shape)}")

    if aux.shape[-1] == target_length:
        return aux
    return F.interpolate(aux, size=target_length, mode="linear", align_corners=False)


class Adapter1d(nn.Module):
    """Project aux features to ``[B, out_dim, target_length]`` via 1x1 Conv1d."""

    def __init__(self, aux_in_dim: int, out_dim: int = 64):
        super().__init__()
        self.aux_in_dim = aux_in_dim
        self.out_dim = out_dim
        self.proj = nn.Conv1d(aux_in_dim, out_dim, kernel_size=1)

    def forward(self, aux: Tensor, target_length: int) -> Tensor:
        x = _prepare_aux(aux, target_length)
        if x.shape[1] != self.aux_in_dim:
            raise ValueError(
                f"Adapter1d expected aux_in_dim={self.aux_in_dim}, got C={x.shape[1]}"
            )
        return self.proj(x)


class AdapterFiLM(nn.Module):

    def __init__(self, aux_in_dim: int, feat_dim: int, gamma_init: float = 0.1):
        super().__init__()
        self.aux_in_dim = aux_in_dim
        self.feat_dim = feat_dim
        self.gamma_init = float(gamma_init)

        self.proj = nn.Conv1d(aux_in_dim, feat_dim * 2, kernel_size=1)

        fan_in = max(aux_in_dim, 1)
        gamma_std = self.gamma_init / math.sqrt(fan_in)
        with torch.no_grad():
            nn.init.normal_(self.proj.weight[:feat_dim], mean=0.0, std=gamma_std)
            nn.init.zeros_(self.proj.weight[feat_dim:])
            nn.init.zeros_(self.proj.bias)

    def forward(self, aux: Tensor, feat: Tensor) -> Tensor:
        if feat.ndim != 3:
            raise ValueError(f"feat must be [B, D, T], got shape={tuple(feat.shape)}")
        if feat.shape[1] != self.feat_dim:
            raise ValueError(f"AdapterFiLM expected feat_dim={self.feat_dim}, got {feat.shape[1]}")

        gb = self.proj(_prepare_aux(aux, feat.shape[-1]))
        gamma, beta = gb.chunk(2, dim=1)
        return (1.0 + gamma) * feat + beta
