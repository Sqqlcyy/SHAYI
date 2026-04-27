from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ae_variant_b_codec.codec_bridge import CodecBridgeConfig, SeanetSmokeConfig, _CodecBridge
from losses.grl_adv import PitchAdversary
from models.heads import (
    AuxFiLMInject,
    PitchEmbed,
    PitchHeadToeplitz,
    RhythmHeadConv,
    RhythmScalarProjector,
    TimbreBrightnessProjector,
    TimbreFiLMGen,
    TimbreHeadGlobal,
)
from models.trunks import Trunk, TrunkConfig


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class VariantBCodecConfig:
    # ---- Codec bridge (frozen DAC by default) ----
    codec_backend: str = "dac"                  # "dac" | "seanet"
    codec_model: str = "dac_44khz"              # see codec_bridge._DAC_MODEL_ALIASES
    codec_bitrate: str = "8kbps"                # DAC bitrate: "8kbps" | "16kbps"
    codec_dim: int = 0                          # 0 = auto-infer (mapped to None for bridge)
    codec_sample_rate: int = 44100
    freeze_codec: bool = True

    # ---- SEANet smoke fallback (only active when codec_backend='seanet') ----
    in_channels: int = 1
    codec_ratios: List[int] = field(default_factory=lambda: [8, 8, 8])
    codec_n_filters: int = 32
    codec_n_residual_layers: int = 3
    smoke_codec_dim: int = 256
    norm: str = "weight_norm"
    causal: bool = False

    # ---- Aux input dims ----
    aux_pitch_dim: int = 588                    # CQT salience bins
    aux_rhythm_dim: int = 8                     # onset_env + onset_map + 6 tempo bands
    aux_timbre_dim: int = 80                    # CheapTrick envelope mel dim
    aux_zero_mask: List[str] = field(default_factory=list)

    # ---- Per-head trunk projectors ----
    trunk_out_p: int = 256
    trunk_out_r: int = 128
    trunk_out_t: int = 256
    trunk_hidden: int = 256
    trunk_n_layers: int = 3
    trunk_kernel: int = 3
    no_trunk: bool = False

    # ---- Structured heads ----
    K: int = 4
    n_pitch_bins: int = 128
    d_r: int = 64
    d_t: int = 128
    head_hidden: int = 256
    aux_film_gamma_init: float = 0.1
    pitch_head_use_toeplitz: bool = True
    pitch_conv_out_gain: float = 5.0

    # ---- FiLM fusion decoder ----
    film_fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 1024])
    pitch_embed_dim: int = 128
    pitch_embed_hidden: int = 256
    film_gamma_clamp: float = 0.5
    film_zt_mode: str = "film"                  # "film" | "concat"

    # ---- Anti-leak secondary defenses (off by default) ----
    use_pitch_adversary: bool = False
    pitch_adversary_lambd: float = 0.1
    pitch_adversary_hidden: int = 256
    z_t_noise_std: float = 0.0

    def to_codec_bridge_cfg(self) -> CodecBridgeConfig:
        return CodecBridgeConfig(
            codec_backend=self.codec_backend,
            codec_model=self.codec_model,
            codec_bitrate=self.codec_bitrate,
            codec_dim=(self.codec_dim if self.codec_dim and self.codec_dim > 0 else None),
            codec_sample_rate=self.codec_sample_rate,
            freeze_codec=self.freeze_codec,
            seanet=SeanetSmokeConfig(
                in_channels=self.in_channels,
                codec_dim=self.smoke_codec_dim,
                ratios=list(self.codec_ratios),
                n_filters=self.codec_n_filters,
                n_residual_layers=self.codec_n_residual_layers,
                norm=self.norm,
                causal=self.causal,
            ),
        )


# ===========================================================================
# FiLM Fusion
# ===========================================================================
class FiLMFusion(nn.Module):

    def __init__(
        self,
        pitch_embed_dim: int,
        d_r: int,
        d_t: int,
        codec_dim: int,
        hidden_dims: List[int],
        gamma_init: float = 0.1,
        gamma_clamp: float = 0.5,
        zt_mode: str = "film",
    ):
        super().__init__()
        if zt_mode not in ("film", "concat"):
            raise ValueError(f"FiLMFusion: zt_mode must be 'film' or 'concat', got {zt_mode!r}")
        self.zt_mode = zt_mode
        in_dim = pitch_embed_dim + d_r + (d_t if zt_mode == "concat" else 0)
        if zt_mode == "film":
            self.film_gen = TimbreFiLMGen(
                d_t=d_t,
                hidden_dims=list(hidden_dims) + [codec_dim],
                gamma_init=gamma_init,
                gamma_clamp=gamma_clamp,
            )
        else:
            self.film_gen = None

        convs = []
        prev = in_dim
        kernels = [1] + [3] * (len(hidden_dims) - 1) + [1]
        kernels = kernels[: len(hidden_dims) + 1]
        while len(kernels) < len(hidden_dims) + 1:
            kernels.append(1)

        for h, k in zip(hidden_dims, kernels):
            convs.append(nn.Conv1d(prev, h, k, padding=k // 2))
            prev = h
        convs.append(nn.Conv1d(prev, codec_dim, kernels[-1], padding=kernels[-1] // 2))
        self.convs = nn.ModuleList(convs)
        self.n_layers = len(hidden_dims) + 1

    def forward(self, pitch_embed: Tensor, z_r: Tensor, z_t: Tensor) -> Tensor:
        if z_r.shape[-1] != pitch_embed.shape[-1]:
            z_r = F.interpolate(z_r, size=pitch_embed.shape[-1], mode="linear", align_corners=False)
        if self.zt_mode == "concat":
            # broadcast z_t [B, d_t] over T → [B, d_t, T]
            z_t_b = z_t.unsqueeze(-1).expand(-1, -1, pitch_embed.shape[-1])
            h = torch.cat([pitch_embed, z_r, z_t_b], dim=1)
            films = []
        else:
            h = torch.cat([pitch_embed, z_r], dim=1)
            films = self.film_gen(z_t)
        for i, conv in enumerate(self.convs):
            h = conv(h)
            if i < len(films):
                gamma, beta = films[i]
                h = (1.0 + gamma) * h + beta
            if i < len(self.convs) - 1:
                h = F.silu(h)
        return h


# ===========================================================================
# VariantBCodec main model (v2)
# ===========================================================================
class VariantBCodec(nn.Module):

    input_mode = "waveform"
    uses_aux_adapters = True

    def __init__(self, cfg: Optional[VariantBCodecConfig] = None):
        super().__init__()
        cfg = cfg or VariantBCodecConfig()
        self.cfg = cfg

        # Frozen codec bridge
        self.codec_bridge = _CodecBridge(cfg.to_codec_bridge_cfg())
        codec_dim = self.codec_bridge.codec_dim
        self.codec_dim = codec_dim

        # Per-head trunks (bottleneck).
        trunk_n_layers = 1 if cfg.no_trunk else cfg.trunk_n_layers
        self.trunk_p = Trunk(TrunkConfig(
            in_dim=codec_dim, out_dim=cfg.trunk_out_p,
            hidden_dim=cfg.trunk_hidden, n_layers=trunk_n_layers,
            kernel=cfg.trunk_kernel,
        ))
        self.trunk_r = Trunk(TrunkConfig(
            in_dim=codec_dim, out_dim=cfg.trunk_out_r,
            hidden_dim=cfg.trunk_hidden, n_layers=trunk_n_layers,
            kernel=cfg.trunk_kernel,
        ))
        self.trunk_t = Trunk(TrunkConfig(
            in_dim=codec_dim, out_dim=cfg.trunk_out_t,
            hidden_dim=cfg.trunk_hidden, n_layers=trunk_n_layers,
            kernel=cfg.trunk_kernel,
        ))

        # Aux FiLM injection per head
        self.aux_p = AuxFiLMInject(cfg.aux_pitch_dim, cfg.trunk_out_p, gamma_init=cfg.aux_film_gamma_init)
        self.aux_r = AuxFiLMInject(cfg.aux_rhythm_dim, cfg.trunk_out_r, gamma_init=cfg.aux_film_gamma_init)
        self.aux_t = AuxFiLMInject(cfg.aux_timbre_dim, cfg.trunk_out_t, gamma_init=cfg.aux_film_gamma_init)

        # Structured heads
        self.head_p = PitchHeadToeplitz(
            feat_dim=cfg.trunk_out_p,
            K=cfg.K,
            n_pitch_bins=cfg.n_pitch_bins,
            hidden=cfg.head_hidden,
            use_toeplitz=cfg.pitch_head_use_toeplitz,
            conv_out_gain=cfg.pitch_conv_out_gain,
        )
        self.head_r = RhythmHeadConv(
            feat_dim=cfg.trunk_out_r,
            d_r=cfg.d_r,
            hidden=cfg.head_hidden,
        )
        self.head_t = TimbreHeadGlobal(
            feat_dim=cfg.trunk_out_t,
            d_t=cfg.d_t,
            hidden=cfg.head_hidden,
        )

        # Scalar projectors used by Stage 2 equivariance losses
        self.rhythm_scalar_proj = RhythmScalarProjector(d_r=cfg.d_r)
        self.timbre_brightness_proj = TimbreBrightnessProjector(d_t=cfg.d_t)

        # FiLM decoder
        self.pitch_embed = PitchEmbed(
            K=cfg.K, n_pitch_bins=cfg.n_pitch_bins,
            d_out=cfg.pitch_embed_dim, hidden=cfg.pitch_embed_hidden,
        )
        self.decoder_film = FiLMFusion(
            pitch_embed_dim=cfg.pitch_embed_dim,
            d_r=cfg.d_r,
            d_t=cfg.d_t,
            codec_dim=codec_dim,
            hidden_dims=list(cfg.film_fusion_hidden_dims),
            gamma_init=cfg.aux_film_gamma_init,
            gamma_clamp=cfg.film_gamma_clamp,
            zt_mode=cfg.film_zt_mode,
        )

        # Optional pitch adversary on z_t (anti-leak secondary defense)
        self.pitch_adversary: Optional[PitchAdversary] = None
        if cfg.use_pitch_adversary:
            self.pitch_adversary = PitchAdversary(
                latent_dim=cfg.d_t,
                n_pitch_bins=cfg.aux_pitch_dim,
                hidden_dim=cfg.pitch_adversary_hidden,
                lambd=cfg.pitch_adversary_lambd,
            )


    # ------------------------------------------------------------------
    def encode(
        self,
        x: Tensor,
        aux_pitch: Tensor,
        aux_rhythm: Tensor,
        aux_timbre: Tensor,
        *,
        deterministic: bool = False,    # noqa: ARG002 (kept for trainer parity)
    ) -> Dict[str, Tensor]:
        c = self.codec_bridge.encode_latent(x)
        return self.encode_from_codec_latent(
            c, aux_pitch, aux_rhythm, aux_timbre, deterministic=deterministic
        )

    def encode_from_codec_latent(
        self,
        c: Tensor,
        aux_pitch: Tensor,
        aux_rhythm: Tensor,
        aux_timbre: Tensor,
        *,
        deterministic: bool = False,    # noqa: ARG002
    ) -> Dict[str, Tensor]:
        c_p = self.trunk_p(c)
        c_r = self.trunk_r(c)
        c_t = self.trunk_t(c)

        c_p = self.aux_p(c_p, aux_pitch)
        c_r = self.aux_r(c_r, aux_rhythm)
        c_t = self.aux_t(c_t, aux_timbre)

        z_p = self.head_p(c_p)       # [B, K, n_pitch_bins, T]
        z_r = self.head_r(c_r)       # [B, d_r, T]
        z_t = self.head_t(c_t)       # [B, d_t]

        if self.training and self.cfg.z_t_noise_std > 0.0:
            z_t = z_t + torch.randn_like(z_t) * float(self.cfg.z_t_noise_std)

        return {
            "feat": c,
            "codec_target": c.detach(),
            "z_p": z_p,
            "z_r": z_r,
            "z_t": z_t,
        }

    def decode(self, z_p: Tensor, z_r: Tensor, z_t: Tensor, target_T: Optional[int] = None) -> Dict[str, Tensor]:
        T = target_T if target_T is not None else z_p.shape[-1]
        pitch_embed = self.pitch_embed(z_p)
        c_hat = self.decoder_film(pitch_embed, z_r, z_t)
        if c_hat.shape[-1] != T:
            c_hat = F.interpolate(c_hat, size=T, mode="linear", align_corners=False)
        return {"codec_hat": c_hat}

    def forward(
        self,
        x: Tensor,
        aux_pitch: Tensor,
        aux_rhythm: Tensor,
        aux_timbre: Tensor,
        *,
        deterministic: bool = False,
    ) -> Dict[str, Tensor]:
        enc = self.encode(x, aux_pitch, aux_rhythm, aux_timbre, deterministic=deterministic)
        c_target = enc["codec_target"]
        dec = self.decode(enc["z_p"], enc["z_r"], enc["z_t"], target_T=c_target.shape[-1])
        c_hat = dec["codec_hat"]

        x_hat = self.codec_bridge.decode_latent(c_hat)

        out: Dict[str, Tensor] = {**enc, "codec_hat": c_hat, "x_hat": x_hat}

        out["mu_p"] = _z_p_flat(enc["z_p"])
        out["log_var_p"] = torch.zeros_like(out["mu_p"])
        out["mu_r"] = enc["z_r"]
        out["log_var_r"] = torch.zeros_like(enc["z_r"])
        out["mu_t"] = enc["z_t"]
        out["log_var_t"] = torch.zeros_like(enc["z_t"])
        return out

    @torch.no_grad()
    def latent_shapes(self, latent_length: int = 861) -> Dict[str, tuple]:
        device = next(self.parameters()).device
        c = torch.zeros(1, self.codec_dim, latent_length, device=device)
        aux_p = torch.zeros(1, self.cfg.aux_pitch_dim, latent_length, device=device)
        aux_r = torch.zeros(1, self.cfg.aux_rhythm_dim, latent_length, device=device)
        aux_t = torch.zeros(1, self.cfg.aux_timbre_dim, latent_length, device=device)
        enc = self.encode_from_codec_latent(c, aux_p, aux_r, aux_t)
        dec = self.decode(enc["z_p"], enc["z_r"], enc["z_t"], target_T=latent_length)
        return {
            "z_p": tuple(enc["z_p"].shape[1:]),
            "z_r": tuple(enc["z_r"].shape[1:]),
            "z_t": tuple(enc["z_t"].shape[1:]),
            "codec_hat": tuple(dec["codec_hat"].shape[1:]),
        }


def _z_p_flat(z_p: Tensor) -> Tensor:
    """Collapse z_p [B, K, n_bins, T] → [B, K*n_bins, T] for legacy logging."""
    B, K, NB, T = z_p.shape
    return z_p.reshape(B, K * NB, T)
