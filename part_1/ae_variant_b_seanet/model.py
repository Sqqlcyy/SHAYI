from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ae_variant_b_codec.model import VariantBCodec, VariantBCodecConfig
from models.seanet_wrap import build_decoder as build_seanet_decoder


@dataclass
class VariantBSEANetConfig(VariantBCodecConfig):
    """Extends B-Codec config with SEANet decoder parameters."""

    # ---- SEANet waveform decoder ----
    dec_in_dim: int = 256
    dec_n_filters: int = 32
    dec_n_residual_layers: int = 3
    dec_ratios: List[int] = field(default_factory=lambda: [8, 8, 4, 2])
    dec_norm: str = "weight_norm"
    dec_lstm: int = 0
    dec_activation: str = "ELU"
    dec_activation_params: Optional[dict] = None
    dec_final_activation: Optional[str] = None


class VariantBSEANet(VariantBCodec):

    def __init__(self, cfg: Optional[VariantBSEANetConfig] = None):
        cfg = cfg or VariantBSEANetConfig()
        super().__init__(cfg)

        # Projection from FiLMFusion output (codec_dim=1024) to SEANet input
        self.dec_proj = nn.Conv1d(self.codec_dim, cfg.dec_in_dim, 1)
        self.seanet_decoder = build_seanet_decoder(
            out_channels=1,
            dimension=cfg.dec_in_dim,
            n_filters=cfg.dec_n_filters,
            n_residual_layers=cfg.dec_n_residual_layers,
            ratios=list(cfg.dec_ratios),
            norm=cfg.dec_norm,
            lstm=cfg.dec_lstm,
            activation=cfg.dec_activation,
            activation_params=cfg.dec_activation_params,
            final_activation=cfg.dec_final_activation,
        )

    def forward(
        self,
        x: Tensor,
        aux_pitch: Tensor,
        aux_rhythm: Tensor,
        aux_timbre: Tensor,
        *,
        deterministic: bool = False,
    ) -> Dict[str, Tensor]:
        enc = self.encode(x, aux_pitch, aux_rhythm, aux_timbre,
                          deterministic=deterministic)
        c_target = enc["codec_target"]
        dec = self.decode(enc["z_p"], enc["z_r"], enc["z_t"],
                          target_T=c_target.shape[-1])
        c_hat = dec["codec_hat"]

        # SEANet decoder path (trainable, not frozen DAC)
        h = self.dec_proj(c_hat)
        x_hat = self.seanet_decoder(h)

        out: Dict[str, Tensor] = {
            **enc,
            "codec_hat": c_hat,
            "x_hat": x_hat,
        }

        # Dummy mu/log_var for trainer parity (deterministic encoder)
        from ae_variant_b_codec.model import _z_p_flat
        out["mu_p"] = _z_p_flat(enc["z_p"])
        out["log_var_p"] = torch.zeros_like(out["mu_p"])
        out["mu_r"] = enc["z_r"]
        out["log_var_r"] = torch.zeros_like(enc["z_r"])
        out["mu_t"] = enc["z_t"]
        out["log_var_t"] = torch.zeros_like(enc["z_t"])
        return out
