from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TrunkConfig:
    in_dim: int = 1024         # DAC-44k codec_dim
    out_dim: int = 256         # factor-specific trunk output channels
    hidden_dim: int = 256
    n_layers: int = 3          # 2 or 3 — 3 is the default from LOEV++ analogue
    kernel: int = 3
    norm: str = "groupnorm"    # "groupnorm" | "batchnorm" | "none"
    num_groups: int = 8        # for GroupNorm
    activation: str = "silu"


class Trunk(nn.Module):

    def __init__(self, cfg: Optional[TrunkConfig] = None, **kwargs):
        super().__init__()
        cfg = cfg or TrunkConfig(**kwargs)
        self.cfg = cfg

        act_cls = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[cfg.activation]

        def _norm(n_ch: int) -> nn.Module:
            if cfg.norm == "groupnorm":
                g = min(cfg.num_groups, n_ch)
                while n_ch % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, n_ch)
            if cfg.norm == "batchnorm":
                return nn.BatchNorm1d(n_ch)
            return nn.Identity()

        layers: list[nn.Module] = []
        prev = cfg.in_dim
        for i in range(max(cfg.n_layers - 1, 0)):
            layers.append(
                nn.Conv1d(prev, cfg.hidden_dim, kernel_size=cfg.kernel, padding=cfg.kernel // 2)
            )
            layers.append(_norm(cfg.hidden_dim))
            layers.append(act_cls())
            prev = cfg.hidden_dim
        layers.append(nn.Conv1d(prev, cfg.out_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

        self.input_conv = self.net[0]

    def forward(self, c: Tensor) -> Tensor:
        """c: [B, in_dim, T] → [B, out_dim, T]."""
        return self.net(c)

    def input_channel_usage(self) -> Tensor:
        w = self.input_conv.weight  # [hidden, in_dim, k]
        return w.abs().sum(dim=(0, 2))  # [in_dim]

    def input_channel_group_lasso(self) -> Tensor:
        w = self.input_conv.weight  # [hidden, in_dim, k]
        # sqrt(sum W^2) per input channel
        return torch.sqrt((w ** 2).sum(dim=(0, 2)) + 1e-12)  # [in_dim]


def trunk_l1_reg(trunks: Sequence[Trunk], form: str = "l1") -> Tensor:
    if not trunks:
        return torch.zeros((), requires_grad=False)
    if form == "l1":
        penalties = [t.input_channel_usage().sum() for t in trunks]
    elif form == "group_lasso":
        penalties = [t.input_channel_group_lasso().sum() for t in trunks]
    else:
        raise ValueError(f"Unknown trunk_l1 form: {form}")
    return torch.stack(penalties).sum()
