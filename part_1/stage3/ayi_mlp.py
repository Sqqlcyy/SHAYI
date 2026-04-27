from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AYIMLPConfig:
    feature_dim: int = 7
    hidden_dim: int = 256
    n_layers: int = 3
    activation: str = "gelu"  # "gelu" | "relu"
    zero_init_last: bool = False


def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int, act: str, zero_last: bool) -> nn.Sequential:
    if act == "gelu":
        act_cls = nn.GELU
    elif act == "relu":
        act_cls = nn.ReLU
    else:
        raise ValueError(f"Unknown activation: {act}")

    layers: List[nn.Module] = []
    prev = in_dim
    for _ in range(max(n_layers - 1, 0)):
        layers.append(nn.Linear(prev, hidden))
        layers.append(act_cls())
        prev = hidden
    last = nn.Linear(prev, out_dim)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class AYIMLP(nn.Module):

    def __init__(self, cfg: AYIMLPConfig = None):
        super().__init__()
        self.cfg = cfg or AYIMLPConfig()
        self.mlp_pitch = _mlp(self.cfg.feature_dim, 1, self.cfg.hidden_dim, self.cfg.n_layers, self.cfg.activation, self.cfg.zero_init_last)
        self.mlp_rhythm = _mlp(self.cfg.feature_dim, 1, self.cfg.hidden_dim, self.cfg.n_layers, self.cfg.activation, self.cfg.zero_init_last)
        self.mlp_timbre = _mlp(self.cfg.feature_dim, 1, self.cfg.hidden_dim, self.cfg.n_layers, self.cfg.activation, self.cfg.zero_init_last)

    @staticmethod
    def pool_latent(z: Tensor) -> Tensor:
        if z.ndim == 3:
            return z.mean(dim=-1)
        if z.ndim == 2:
            return z
        raise ValueError(f"latent must be [B,D] or [B,D,T], got {tuple(z.shape)}")

    @staticmethod
    def build_features(z_anchor: Tensor, z_transformed: Tensor) -> Tensor:
        u = AYIMLP.pool_latent(z_anchor)
        v = AYIMLP.pool_latent(z_transformed)
        delta = v - u
        cos_uv = F.cosine_similarity(u, v, dim=-1, eps=1e-8).unsqueeze(-1)
        features = torch.cat(
            [
                u.mean(dim=-1, keepdim=True),
                u.std(dim=-1, keepdim=True, unbiased=False),
                v.mean(dim=-1, keepdim=True),
                v.std(dim=-1, keepdim=True, unbiased=False),
                cos_uv,
                delta.abs().mean(dim=-1, keepdim=True),
                delta.norm(p=2, dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        if features.shape[-1] != 7:
            raise RuntimeError(f"Expected 7-D probe features, got {features.shape[-1]}")
        return features

    def forward(
        self,
        phi_pitch: Optional[Tensor] = None,
        phi_rhythm: Optional[Tensor] = None,
        phi_timbre: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        return {
            "pitch": self.mlp_pitch(phi_pitch) if phi_pitch is not None else None,
            "rhythm": self.mlp_rhythm(phi_rhythm) if phi_rhythm is not None else None,
            "timbre": self.mlp_timbre(phi_timbre) if phi_timbre is not None else None,
        }
