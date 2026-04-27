"""Variant A-Codec — independent feature encoders + frozen DAC decoder.

Hybrid of Variant A (physically-separated feature encoders, part1_old lineage)
and Variant B (frozen DAC codec bridge for reconstruction).

Data flow
---------
feat_pitch  [B,588,T]  → Enc_p  → head_p  → z_p  [B,K,128,T]
feat_rhythm [B,8,T]    → Enc_r  → head_r  → z_r  [B,d_r,T]
feat_mel    [B,128,T]  → Enc_c  → c_content [B,H,T]
                                  └──→ (optional) AuxFiLM(envelope) → c_content
                                  └──→ head_t (pool)  → z_t  [B,d_t]

Fusion (same interface as Variant B's FiLMFusion but fed by Variant A heads):
    pitch_emb = PitchEmbed(z_p)
    h = FiLMFusion([pitch_emb, z_r], z_t)    → [B, codec_dim=1024, T]
    h = h + Proj(c_content)                  → residual absorbs phase / transients

Reconstruction:
    c_target = DAC_encoder(x).detach()          (frozen)
    L_codec_distill = MSE(h, c_target)
    x_hat = DAC_decoder(h)                      (frozen)

Anti-leak on the residual (CRITICAL):
    z_content residual can silently hoard pitch/rhythm/timbre info if
    we don't constrain it. Three GRL-adversary heads try to predict
    each factor from a time-pool of c_content; gradient-reversed loss
    pushes c_content AWAY from factor-predictive directions, so the
    residual only carries phase / ambience / non-factor content.

Rationale
---------
Variant B shares a trunk (all three heads see codec c) and relies on
group LASSO + aux FiLM + aux dropout to carve specialized views →
*soft* disentanglement. Variant A-Codec keeps Variant A's **physical**
input separation (each encoder sees a different feature), so cross-factor
leakage is blocked at the input level, not just by losses — disentanglement
is harder. DAC decoder gives us phase-correct audio without training a
vocoder from scratch. The trade-off vs Variant B is harder convergence
(feature → c is a one-to-many regression) and blurrier output unless we
train longer.

Trainer integration
-------------------
``input_mode = "feature"``: trainer passes (feat_pitch, feat_rhythm,
feat_mel) and optionally (feat_envelope) and (aux_pitch, aux_rhythm,
aux_timbre) for the content adversary. The model computes and returns
``loss/content_adv`` as part of the output dict; Stage1Trainer multiplies
by ``w_content_adv`` in ``_compute_losses``.

``require_codec_distill: true`` in yaml: codec_hat and codec_target are
returned so the existing Stage1Trainer codec_distill path activates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ae_variant_b_codec.codec_bridge import CodecBridgeConfig, SeanetSmokeConfig, _CodecBridge
from ae_variant_b_codec.model import FiLMFusion
from ae_variant_a.model import FeatureEncoder
from losses.grl_adv import FactorAdversary, PitchAdversary, gradient_reverse
from models.heads import (
    AuxFiLMInject,
    PitchEmbed,
    PitchHeadToeplitz,
    RhythmHeadConv,
    RhythmScalarProjector,
    TimbreBrightnessProjector,
    TimbreHeadGlobal,
)


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class VariantACodecConfig:
    # ---- Codec bridge (frozen DAC 44.1 kHz) ----
    codec_backend: str = "dac"
    codec_model: str = "dac_44khz"
    codec_bitrate: str = "8kbps"
    codec_dim: int = 0                   # 0 = auto-infer
    codec_sample_rate: int = 44100
    freeze_codec: bool = True
    in_channels: int = 1
    codec_ratios: List[int] = field(default_factory=lambda: [8, 8, 8])
    codec_n_filters: int = 32
    codec_n_residual_layers: int = 3
    smoke_codec_dim: int = 256
    norm: str = "weight_norm"
    causal: bool = False

    # ---- Feature-side encoder inputs ----
    pitch_in_dim: int = 588              # CQT salience bins
    rhythm_in_dim: int = 8               # 03b multi-channel rhythm
    mel_in_dim: int = 128                # log-mel
    envelope_in_dim: int = 80
    use_envelope_aux: bool = True        # FiLM inject envelope into c_content

    # ---- Per-head feature encoders ----
    enc_hidden: int = 256
    enc_n_layers_side: int = 3           # pitch + rhythm (light)
    enc_n_layers_main: int = 5           # content (deeper — main info path)
    enc_kernel: int = 3

    # ---- Structured heads (shapes match Variant B so L_ortho/L_AY reuse) ----
    K: int = 4
    n_pitch_bins: int = 128
    d_r: int = 64
    d_t: int = 128
    head_hidden: int = 256
    pitch_head_use_toeplitz: bool = True
    pitch_conv_out_gain: float = 5.0
    envelope_film_gamma_init: float = 0.1

    # ---- FiLM fusion → codec_dim ----
    film_fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 1024])
    pitch_embed_dim: int = 128
    pitch_embed_hidden: int = 256
    film_gamma_init: float = 0.1
    film_gamma_clamp: float = 0.5
    film_zt_mode: str = "film"           # "film" | "concat"
    content_residual: bool = True
    content_residual_dim: int = 32
    content_stop_grad: bool = True
    content_warmup_steps: int = 3000

    # ---- Content adversary (GRL on z_content) ----
    adv_hidden: int = 256
    adv_lambd: float = 0.5               # gradient-reversal strength
    adv_pitch_target_dim: int = 588
    adv_rhythm_target_dim: int = 8
    adv_timbre_target_dim: int = 80

    # ---- v2: Factor adversaries on z_t (not z_content) ----
    # GRL adversaries try to predict pitch/rhythm from z_t. Gradient reversal
    # pushes z_t away from encoding pitch/rhythm. Orthogonal to the input-side
    # envelope FiLM (which removes pitch at input); this adds an output-side
    # guard that works even if envelope FiLM leaks some.
    use_adv_on_zt: bool = False
    adv_pitch_on_zt_lambd: float = 0.3
    adv_rhythm_on_zt_lambd: float = 0.3

    # Kept for trainer parity (aux_dropout hooks read this).
    aux_zero_mask: Tuple[str, ...] = ()


# GRL adversary classes moved to losses/grl_adv.py. 


# ===========================================================================
# Main model
# ===========================================================================
class VariantACodec(nn.Module):
    # Trainer contract
    input_mode = "feature"
    uses_aux_adapters = False           # not Variant B shared-trunk
    needs_target_wav = False

    def __init__(self, cfg: Optional[VariantACodecConfig] = None):
        super().__init__()
        cfg = cfg or VariantACodecConfig()
        self.cfg = cfg

        # Frozen DAC bridge
        self.codec_bridge = _CodecBridge(self._bridge_cfg())
        codec_dim = self.codec_bridge.codec_dim
        self.codec_dim = codec_dim

        # Three independent feature encoders
        self.enc_pitch = FeatureEncoder(
            in_dim=cfg.pitch_in_dim, hidden=cfg.enc_hidden,
            n_layers=cfg.enc_n_layers_side, kernel=cfg.enc_kernel,
        )
        self.enc_rhythm = FeatureEncoder(
            in_dim=cfg.rhythm_in_dim, hidden=cfg.enc_hidden,
            n_layers=cfg.enc_n_layers_side, kernel=cfg.enc_kernel,
        )
        self.enc_content = FeatureEncoder(
            in_dim=cfg.mel_in_dim, hidden=cfg.enc_hidden,
            n_layers=cfg.enc_n_layers_main, kernel=cfg.enc_kernel,
        )

        # Optional envelope FiLM on content encoder output
        self.aux_envelope: Optional[AuxFiLMInject] = None
        if cfg.use_envelope_aux:
            self.aux_envelope = AuxFiLMInject(
                aux_dim=cfg.envelope_in_dim,
                feat_dim=cfg.enc_hidden,
                gamma_init=cfg.envelope_film_gamma_init,
            )

        # Structured heads — same shapes as Variant B
        self.head_p = PitchHeadToeplitz(
            feat_dim=cfg.enc_hidden,
            K=cfg.K,
            n_pitch_bins=cfg.n_pitch_bins,
            hidden=cfg.head_hidden,
            use_toeplitz=cfg.pitch_head_use_toeplitz,
            conv_out_gain=cfg.pitch_conv_out_gain,
        )
        self.head_r = RhythmHeadConv(
            feat_dim=cfg.enc_hidden, d_r=cfg.d_r, hidden=cfg.head_hidden,
        )
        self.head_t = TimbreHeadGlobal(
            feat_dim=cfg.enc_hidden, d_t=cfg.d_t, hidden=cfg.head_hidden,
        )

        # Stage 2 scalar projectors (needed by L_AY if/when Stage 2 runs)
        self.rhythm_scalar_proj = RhythmScalarProjector(d_r=cfg.d_r)
        self.timbre_brightness_proj = TimbreBrightnessProjector(d_t=cfg.d_t)

        # Fusion → codec_dim (reuse Variant B's FiLMFusion)
        self.pitch_embed = PitchEmbed(
            K=cfg.K, n_pitch_bins=cfg.n_pitch_bins,
            d_out=cfg.pitch_embed_dim, hidden=cfg.pitch_embed_hidden,
        )
        self.fusion = FiLMFusion(
            pitch_embed_dim=cfg.pitch_embed_dim,
            d_r=cfg.d_r,
            d_t=cfg.d_t,
            codec_dim=codec_dim,
            hidden_dims=list(cfg.film_fusion_hidden_dims),
            gamma_init=cfg.film_gamma_init,
            gamma_clamp=cfg.film_gamma_clamp,
            zt_mode=cfg.film_zt_mode,
        )
        self.content_bottleneck = None
        self.content_proj = None
        if cfg.content_residual:
            self.content_bottleneck = nn.Conv1d(cfg.enc_hidden, cfg.content_residual_dim, 1)
            self.content_proj = nn.Conv1d(cfg.content_residual_dim, codec_dim, 1)
        self._global_step = 0

        # Content adversaries (GRL) — use FactorAdversary from losses/grl_adv.
        self.adv_pitch = FactorAdversary(
            latent_dim=cfg.content_residual_dim, target_dim=cfg.adv_pitch_target_dim,
            hidden=cfg.adv_hidden, lambd=cfg.adv_lambd,
        )
        self.adv_rhythm = FactorAdversary(
            latent_dim=cfg.content_residual_dim, target_dim=cfg.adv_rhythm_target_dim,
            hidden=cfg.adv_hidden, lambd=cfg.adv_lambd,
        )
        self.adv_timbre = FactorAdversary(
            latent_dim=cfg.content_residual_dim, target_dim=cfg.adv_timbre_target_dim,
            hidden=cfg.adv_hidden, lambd=cfg.adv_lambd,
        )

        # v2: factor adversaries on z_t (output-side anti-leak).
        self.adv_pitch_on_zt: Optional[PitchAdversary] = None
        self.adv_rhythm_on_zt: Optional[PitchAdversary] = None
        if cfg.use_adv_on_zt:
            # PitchAdversary is generic: predicts target_dim from latent via GRL.
            self.adv_pitch_on_zt = PitchAdversary(
                latent_dim=cfg.d_t,
                n_pitch_bins=cfg.adv_pitch_target_dim,
                hidden_dim=cfg.adv_hidden,
                lambd=cfg.adv_pitch_on_zt_lambd,
            )
            self.adv_rhythm_on_zt = PitchAdversary(
                latent_dim=cfg.d_t,
                n_pitch_bins=cfg.adv_rhythm_target_dim,
                hidden_dim=cfg.adv_hidden,
                lambd=cfg.adv_rhythm_on_zt_lambd,
            )

    # ------------------------------------------------------------------
    def _bridge_cfg(self) -> CodecBridgeConfig:
        cfg = self.cfg
        return CodecBridgeConfig(
            codec_backend=cfg.codec_backend,
            codec_model=cfg.codec_model,
            codec_bitrate=cfg.codec_bitrate,
            codec_dim=(cfg.codec_dim if cfg.codec_dim > 0 else None),
            codec_sample_rate=cfg.codec_sample_rate,
            freeze_codec=cfg.freeze_codec,
            seanet=SeanetSmokeConfig(
                in_channels=cfg.in_channels,
                codec_dim=cfg.smoke_codec_dim,
                ratios=list(cfg.codec_ratios),
                n_filters=cfg.codec_n_filters,
                n_residual_layers=cfg.codec_n_residual_layers,
                norm=cfg.norm,
                causal=cfg.causal,
            ),
        )

    @property
    def uses_envelope_aux(self) -> bool:
        return self.aux_envelope is not None

    def encode_only(
        self,
        feat_pitch: Optional[Tensor],
        feat_rhythm: Optional[Tensor],
        feat_mel: Optional[Tensor],
        feat_envelope: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Encoder + heads only — NO DAC decoder / codec_distill. Used by
        Stage 2 L_AY to avoid OOM on the transformed forward pass."""
        T = min(feat_pitch.shape[-1], feat_rhythm.shape[-1], feat_mel.shape[-1])
        feat_pitch = feat_pitch[..., :T]
        feat_rhythm = feat_rhythm[..., :T]
        feat_mel = feat_mel[..., :T]

        c_p = self.enc_pitch(feat_pitch)
        c_r = self.enc_rhythm(feat_rhythm)
        c_content = self.enc_content(feat_mel)

        if self.aux_envelope is not None and feat_envelope is not None:
            env = feat_envelope
            if env.shape[-1] != c_content.shape[-1]:
                env = F.interpolate(env, size=c_content.shape[-1], mode="linear", align_corners=False)
            c_content = self.aux_envelope(c_content, env)

        z_p = self.head_p(c_p)
        z_r = self.head_r(c_r)
        z_t = self.head_t(c_content)
        z_content_out = self.content_bottleneck(c_content) if self.content_bottleneck is not None else c_content
        return {"z_p": z_p, "z_r": z_r, "z_t": z_t, "z_content": z_content_out}

    # ------------------------------------------------------------------
    def forward(
        self,
        feat_pitch: Tensor,                 # [B, pitch_in_dim, T]
        feat_rhythm: Tensor,                # [B, rhythm_in_dim, T]
        feat_mel: Tensor,                   # [B, mel_in_dim, T]
        feat_envelope: Optional[Tensor] = None,  # [B, envelope_in_dim, T]
        *,
        # Aux targets for the GRL content adversary. Trainer routes them in.
        aux_pitch: Optional[Tensor] = None,   # [B, 588, T]
        aux_rhythm: Optional[Tensor] = None,  # [B, 8, T]
        aux_timbre: Optional[Tensor] = None,  # [B, 80, T]
        audio: Optional[Tensor] = None,       # [B, 1, T_audio] — for codec_distill target
        deterministic: bool = False,          # noqa: ARG002
    ) -> Dict[str, Tensor]:
        T = min(feat_pitch.shape[-1], feat_rhythm.shape[-1], feat_mel.shape[-1])
        feat_pitch = feat_pitch[..., :T]
        feat_rhythm = feat_rhythm[..., :T]
        feat_mel = feat_mel[..., :T]

        c_p = self.enc_pitch(feat_pitch)
        c_r = self.enc_rhythm(feat_rhythm)
        c_content = self.enc_content(feat_mel)

        if self.aux_envelope is not None and feat_envelope is not None:
            env = feat_envelope
            if env.shape[-1] != c_content.shape[-1]:
                env = F.interpolate(env, size=c_content.shape[-1], mode="linear", align_corners=False)
            c_content = self.aux_envelope(c_content, env)

        z_p = self.head_p(c_p)                     # [B, K, n_pitch_bins, T]
        z_r = self.head_r(c_r)                     # [B, d_r, T]
        z_t = self.head_t(c_content)               # [B, d_t]

        # Fusion → codec_hat
        pitch_emb = self.pitch_embed(z_p)          # [B, pitch_embed_dim, T]
        h = self.fusion(pitch_emb, z_r, z_t)       # [B, codec_dim, T]
        if self.content_bottleneck is not None:
            z_content_bn = self.content_bottleneck(c_content)
            content_active = (not self.training) or (self._global_step >= self.cfg.content_warmup_steps)
            if content_active:
                residual = z_content_bn.detach() if self.cfg.content_stop_grad else z_content_bn
                h = h + self.content_proj(residual)

        codec_hat = h

        # Codec target (frozen DAC encoder) for L_codec_distill
        codec_target = None
        if audio is not None:
            with torch.no_grad():
                codec_target = self.codec_bridge.encode_latent(audio).detach()

        # Waveform out (frozen DAC decoder)
        x_hat = self.codec_bridge.decode_latent(codec_hat)

        # Content adversary loss (GRL pushes c_content AWAY from factor info)
        content_adv_loss = codec_hat.new_zeros(())
        adv_terms = 0
        if aux_pitch is not None or aux_rhythm is not None or aux_timbre is not None:
            z_content_pool = (z_content_bn if self.content_bottleneck is not None else c_content).mean(dim=-1)        # [B, H]
            if aux_pitch is not None:
                content_adv_loss = content_adv_loss + self.adv_pitch(
                    z_content_pool, aux_pitch.mean(dim=-1),
                )
                adv_terms += 1
            if aux_rhythm is not None:
                content_adv_loss = content_adv_loss + self.adv_rhythm(
                    z_content_pool, aux_rhythm.mean(dim=-1),
                )
                adv_terms += 1
            if aux_timbre is not None:
                content_adv_loss = content_adv_loss + self.adv_timbre(
                    z_content_pool, aux_timbre.mean(dim=-1),
                )
                adv_terms += 1
            if adv_terms > 0:
                content_adv_loss = content_adv_loss / adv_terms

        # v2: GRL factor adversaries on z_t (output-side anti-leak)
        adv_pitch_on_zt_loss = codec_hat.new_zeros(())
        adv_rhythm_on_zt_loss = codec_hat.new_zeros(())
        if self.adv_pitch_on_zt is not None and aux_pitch is not None:
            adv_pitch_on_zt_loss = self.adv_pitch_on_zt(z_t, aux_pitch)
        if self.adv_rhythm_on_zt is not None and aux_rhythm is not None:
            adv_rhythm_on_zt_loss = self.adv_rhythm_on_zt(z_t, aux_rhythm)

        zero = lambda t: t.new_zeros(t.shape)          # noqa: E731
        out: Dict[str, Tensor] = {
            "x_hat": x_hat,
            "z_p": z_p,
            "z_r": z_r,
            "z_t": z_t,
            "z_content": z_content_bn if self.content_bottleneck is not None else c_content,
            "mu_p": z_p, "log_var_p": zero(z_p),
            "mu_r": z_r, "log_var_r": zero(z_r),
            "mu_t": z_t, "log_var_t": zero(z_t),
            "codec_hat": codec_hat,
            "loss/content_adv": content_adv_loss,
            "loss/adv_pitch_on_zt": adv_pitch_on_zt_loss,
            "loss/adv_rhythm_on_zt": adv_rhythm_on_zt_loss,
        }
        if codec_target is not None:
            out["codec_target"] = codec_target
        return out
