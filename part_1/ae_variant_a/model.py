from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from losses.grl_adv import FactorAdversary, PitchAdversary, gradient_reverse  # noqa: F401
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
from models.seanet_wrap import build_decoder


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class VariantAConfig:
    # ---- Feature-side input dims (match dataset) ----
    pitch_in_dim: int = 588          # CQT salience bins (feat_pitch)
    rhythm_in_dim: int = 1           # legacy onset map (feat_rhythm)
    mel_in_dim: int = 128            # log-mel (feat_mel)

    # ---- Per-head feature encoder (no time downsampling) ----
    enc_hidden: int = 256
    enc_n_layers_side: int = 3       # pitch + rhythm encoders (small)
    enc_n_layers_main: int = 5       # content encoder (deeper)
    enc_kernel: int = 3

    # ---- Structured heads (match VariantB's latent shapes for loss parity) ----
    K: int = 4
    n_pitch_bins: int = 128
    d_r: int = 64
    d_t: int = 128
    head_hidden: int = 256
    pitch_head_use_toeplitz: bool = True
    pitch_conv_out_gain: float = 5.0

    # ---- Fusion → decoder input ----
    pitch_embed_dim: int = 128
    pitch_embed_hidden: int = 256
    dec_in_dim: int = 256            # fusion output channels → decoder input
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 512])
    film_gamma_init: float = 0.1
    film_gamma_clamp: float = 0.5
    content_residual: bool = True    # add Proj(z_content) to fusion output
    content_residual_dim: int = 32   # bottleneck dim
    content_stop_grad: bool = True   # detach z_content before decoder (recon grad → factors only)
    content_warmup_steps: int = 3000 # content residual OFF for first N steps

    # ---- SEANet waveform decoder ----
    dec_n_filters: int = 32
    dec_n_residual_layers: int = 3
    dec_ratios: Tuple[int, ...] = (8, 8, 4, 2)   # ∏=512 → 86fps latent → 44.1kHz wav
    dec_norm: str = "weight_norm"
    dec_lstm: int = 0
    dec_final_activation: Optional[str] = None
    dec_activation: str = "ELU"                     # "ELU" | "Snake" | "SnakeBeta"
    dec_activation_params: Optional[dict] = None   # e.g. {"alpha_init": 1.0}

    # ---- Option B: envelope FiLM aux into content encoder ----
    # When True, the content encoder output is modulated by a FiLM layer driven by envelope features (pitch-invariant liftered spectrogram).
    use_envelope_aux: bool = False
    envelope_in_dim: int = 80
    envelope_film_gamma_init: float = 0.1

    # ---- v2: factor adversaries ---
    # All use PitchAdversary class (generic latent → target MSE with GRL).
    # Target dims match aux feature dims from dataset.
    adv_hidden: int = 256
    adv_pitch_target_dim: int = 588
    adv_rhythm_target_dim: int = 8
    adv_envelope_target_dim: int = 80

    # adversaries on z_t — push pitch/rhythm OUT of timbre latent
    use_adv_on_zt: bool = False
    adv_pitch_on_zt_lambd: float = 0.3
    adv_rhythm_on_zt_lambd: float = 0.3

    # adversaries on z_r — push pitch/timbre OUT of rhythm latent
    use_adv_on_zr: bool = False
    adv_pitch_on_zr_lambd: float = 0.3
    adv_envelope_on_zr_lambd: float = 0.3

    # content adversary (on z_content residual) — 3 GRL heads for pitch/rhythm/envelope
    use_content_adv: bool = False
    content_adv_lambd: float = 0.5

    # Kept for trainer parity (aux_dropout hooks look it up); unused here.
    aux_zero_mask: Tuple[str, ...] = ()


# ===========================================================================
# Feature encoder (dilated TCN)
# ===========================================================================
class FeatureEncoder(nn.Module):
    """Stack of 1D conv residual blocks that preserves time length.

    Features arrive at ~86 fps (same rate the SEANet decoder expects at its
    input), so the encoder only needs to widen channels + mix context, not
    change the time axis.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int, kernel: int = 3):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden, kernel, padding=kernel // 2)
        blocks = []
        for i in range(n_layers):
            dilation = 2 ** (i % 4)                 # 1, 2, 4, 8, 1, 2, ...
            pad = dilation * (kernel // 2)
            blocks.append(
                nn.Sequential(
                    nn.GroupNorm(min(8, hidden), hidden),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(hidden, hidden, kernel, padding=pad, dilation=dilation),
                    nn.GroupNorm(min(8, hidden), hidden),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(hidden, hidden, 1),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = h + blk(h)                          # residual
        return h


# ===========================================================================
# FiLM fusion
# ===========================================================================
class VariantAFusion(nn.Module):
    def __init__(
        self,
        pitch_embed_dim: int,
        d_r: int,
        d_t: int,
        out_dim: int,
        hidden_dims: List[int],
        gamma_init: float = 0.1,
        gamma_clamp: float = 0.5,
    ):
        super().__init__()
        in_dim = pitch_embed_dim + d_r
        self.film_gen = TimbreFiLMGen(
            d_t=d_t,
            hidden_dims=list(hidden_dims) + [out_dim],
            gamma_init=gamma_init,
            gamma_clamp=gamma_clamp,
        )
        convs = []
        prev = in_dim
        for h in hidden_dims:
            convs.append(nn.Conv1d(prev, h, 3, padding=1))
            prev = h
        convs.append(nn.Conv1d(prev, out_dim, 1))
        self.convs = nn.ModuleList(convs)

    def forward(self, pitch_embed: Tensor, z_r: Tensor, z_t: Tensor) -> Tensor:
        if z_r.shape[-1] != pitch_embed.shape[-1]:
            z_r = F.interpolate(z_r, size=pitch_embed.shape[-1], mode="linear", align_corners=False)
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
# Main model
# ===========================================================================
class VariantAMultiHead(nn.Module):
    # Trainer contract
    input_mode = "feature"
    uses_aux_adapters = False
    needs_target_wav = False

    def __init__(self, cfg: Optional[VariantAConfig] = None):
        super().__init__()
        cfg = cfg or VariantAConfig()
        self.cfg = cfg

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

        # Structured heads (reused from models.heads → same loss plumbing as VariantB)
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

        # Scalar projectors used by Stage 2 equivariance losses
        self.rhythm_scalar_proj = RhythmScalarProjector(d_r=cfg.d_r)
        self.timbre_brightness_proj = TimbreBrightnessProjector(d_t=cfg.d_t)

        # Option B: envelope FiLM aux for content encoder output.
        # Drives z_t toward pitch-invariance by modulating c_content with
        # the liftered spectral envelope before the timbre pool.
        self.aux_envelope: Optional[AuxFiLMInject] = None
        if cfg.use_envelope_aux:
            self.aux_envelope = AuxFiLMInject(
                aux_dim=cfg.envelope_in_dim,
                feat_dim=cfg.enc_hidden,
                gamma_init=cfg.envelope_film_gamma_init,
            )

        # v2 factor adversaries (output-side anti-leak via GRL)
        # z_t → pitch/rhythm (push timbre away from pitch + rhythm)
        self.adv_pitch_on_zt: Optional[PitchAdversary] = None
        self.adv_rhythm_on_zt: Optional[PitchAdversary] = None
        if cfg.use_adv_on_zt:
            self.adv_pitch_on_zt = PitchAdversary(
                latent_dim=cfg.d_t, n_pitch_bins=cfg.adv_pitch_target_dim,
                hidden_dim=cfg.adv_hidden, lambd=cfg.adv_pitch_on_zt_lambd,
            )
            self.adv_rhythm_on_zt = PitchAdversary(
                latent_dim=cfg.d_t, n_pitch_bins=cfg.adv_rhythm_target_dim,
                hidden_dim=cfg.adv_hidden, lambd=cfg.adv_rhythm_on_zt_lambd,
            )
        # z_r → pitch/envelope (push rhythm away from pitch + timbre)
        self.adv_pitch_on_zr: Optional[PitchAdversary] = None
        self.adv_envelope_on_zr: Optional[PitchAdversary] = None
        if cfg.use_adv_on_zr:
            self.adv_pitch_on_zr = PitchAdversary(
                latent_dim=cfg.d_r, n_pitch_bins=cfg.adv_pitch_target_dim,
                hidden_dim=cfg.adv_hidden, lambd=cfg.adv_pitch_on_zr_lambd,
            )
            self.adv_envelope_on_zr = PitchAdversary(
                latent_dim=cfg.d_r, n_pitch_bins=cfg.adv_envelope_target_dim,
                hidden_dim=cfg.adv_hidden, lambd=cfg.adv_envelope_on_zr_lambd,
            )
        # z_content → pitch/rhythm/envelope (residual anti-leak)
        self.content_adv_pitch: Optional[PitchAdversary] = None
        self.content_adv_rhythm: Optional[PitchAdversary] = None
        self.content_adv_envelope: Optional[PitchAdversary] = None
        if cfg.use_content_adv:
            self.content_adv_pitch = PitchAdversary(
                latent_dim=cfg.content_residual_dim, n_pitch_bins=cfg.adv_pitch_target_dim,
                hidden_dim=cfg.adv_hidden, lambd=cfg.content_adv_lambd,
            )
            self.content_adv_rhythm = PitchAdversary(
                latent_dim=cfg.content_residual_dim, n_pitch_bins=cfg.adv_rhythm_target_dim,
                hidden_dim=cfg.adv_hidden, lambd=cfg.content_adv_lambd,
            )
            self.content_adv_envelope = PitchAdversary(
                latent_dim=cfg.content_residual_dim, n_pitch_bins=cfg.adv_envelope_target_dim,
                hidden_dim=cfg.adv_hidden, lambd=cfg.content_adv_lambd,
            )

        # Fusion → decoder input space
        self.pitch_embed = PitchEmbed(
            K=cfg.K, n_pitch_bins=cfg.n_pitch_bins,
            d_out=cfg.pitch_embed_dim, hidden=cfg.pitch_embed_hidden,
        )
        self.fusion = VariantAFusion(
            pitch_embed_dim=cfg.pitch_embed_dim,
            d_r=cfg.d_r,
            d_t=cfg.d_t,
            out_dim=cfg.dec_in_dim,
            hidden_dims=list(cfg.fusion_hidden_dims),
            gamma_init=cfg.film_gamma_init,
            gamma_clamp=cfg.film_gamma_clamp,
        )
        # Optional content residual projection (z_content → dec_in space).
        # Content residual: enc_hidden → bottleneck (32) → dec_in_dim.
        # Small bottleneck prevents z_content from bypassing factor latents.
        # stop-gradient ensures recon gradient flows only through z_p/z_r/z_t.
        self.content_bottleneck = None
        self.content_proj = None
        if cfg.content_residual:
            self.content_bottleneck = nn.Conv1d(cfg.enc_hidden, cfg.content_residual_dim, 1)
            self.content_proj = nn.Conv1d(cfg.content_residual_dim, cfg.dec_in_dim, 1)
        self._global_step = 0  # set by trainer for content warmup curriculum

        self.decoder = build_decoder(
            out_channels=1,
            dimension=cfg.dec_in_dim,
            n_filters=cfg.dec_n_filters,
            n_residual_layers=cfg.dec_n_residual_layers,
            ratios=list(cfg.dec_ratios),
            causal=False,
            norm=cfg.dec_norm,
            lstm=cfg.dec_lstm,
            final_activation=cfg.dec_final_activation,
            activation=cfg.dec_activation,
            activation_params=cfg.dec_activation_params,
        )

    # ------------------------------------------------------------------
    @property
    def uses_envelope_aux(self) -> bool:
        return self.aux_envelope is not None

    @property
    def uses_adv_kwargs(self) -> bool:
        """True if trainer should route aux_pitch/rhythm/timbre kwargs in."""
        return (
            self.cfg.use_adv_on_zt
            or self.cfg.use_adv_on_zr
            or self.cfg.use_content_adv
        )

    # ------------------------------------------------------------------
    def encode_only(
        self,
        feat_pitch: Optional[Tensor],
        feat_rhythm: Optional[Tensor],
        feat_mel: Optional[Tensor],
        feat_envelope: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Encoder + heads only — NO decoder. Used by Stage 2 L_AY for
        transformed forward (avoids wasting memory on unused x_hat)."""
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
        feat_pitch: Tensor,              # [B, pitch_in_dim, T]
        feat_rhythm: Tensor,             # [B, rhythm_in_dim, T]
        feat_mel: Tensor,                # [B, mel_in_dim, T]
        feat_envelope: Optional[Tensor] = None,   # [B, envelope_in_dim, T]
        *,
        aux_pitch: Optional[Tensor] = None,    # [B, 588, T]
        aux_rhythm: Optional[Tensor] = None,   # [B, 8, T]
        aux_timbre: Optional[Tensor] = None,   # [B, 80, T] (envelope)
        deterministic: bool = False,    # noqa: ARG002
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

        z_p = self.head_p(c_p)
        z_r = self.head_r(c_r)
        z_t = self.head_t(c_content)

        pitch_emb = self.pitch_embed(z_p)
        h = self.fusion(pitch_emb, z_r, z_t)
        if self.content_bottleneck is not None:
            z_content_bn = self.content_bottleneck(c_content)  # [B, 32, T] bottleneck
            # Content warmup: OFF for first N steps → factors must carry all info
            content_active = (not self.training) or (self._global_step >= self.cfg.content_warmup_steps)
            if content_active:
                residual = z_content_bn.detach() if self.cfg.content_stop_grad else z_content_bn
                h = h + self.content_proj(residual)
        x_hat = self.decoder(h)

        # ---- v2 factor adversaries (GRL) ----
        zero_scalar = lambda: x_hat.new_zeros(())  # noqa: E731
        adv_pitch_on_zt = zero_scalar()
        adv_rhythm_on_zt = zero_scalar()
        adv_pitch_on_zr = zero_scalar()
        adv_envelope_on_zr = zero_scalar()
        content_adv_loss = zero_scalar()
        if self.adv_pitch_on_zt is not None and aux_pitch is not None:
            adv_pitch_on_zt = self.adv_pitch_on_zt(z_t, aux_pitch)
        if self.adv_rhythm_on_zt is not None and aux_rhythm is not None:
            adv_rhythm_on_zt = self.adv_rhythm_on_zt(z_t, aux_rhythm)
        if self.adv_pitch_on_zr is not None and aux_pitch is not None:
            adv_pitch_on_zr = self.adv_pitch_on_zr(z_r, aux_pitch)
        if self.adv_envelope_on_zr is not None and aux_timbre is not None:
            adv_envelope_on_zr = self.adv_envelope_on_zr(z_r, aux_timbre)
        if self.content_adv_pitch is not None:
            z_content_pool = (z_content_bn if self.content_bottleneck is not None else c_content).mean(dim=-1)
            terms = []
            if aux_pitch is not None:
                terms.append(self.content_adv_pitch(z_content_pool, aux_pitch.mean(dim=-1)))
            if aux_rhythm is not None:
                terms.append(self.content_adv_rhythm(z_content_pool, aux_rhythm.mean(dim=-1)))
            if aux_timbre is not None:
                terms.append(self.content_adv_envelope(z_content_pool, aux_timbre.mean(dim=-1)))
            if terms:
                content_adv_loss = sum(terms) / len(terms)

        zero_log_var = lambda t: t.new_zeros(t.shape)  # noqa: E731
        return {
            "x_hat": x_hat,
            "z_p": z_p,
            "z_r": z_r,
            "z_t": z_t,
            "z_content": z_content_bn if self.content_bottleneck is not None else c_content,
            "mu_p": z_p, "log_var_p": zero_log_var(z_p),
            "mu_r": z_r, "log_var_r": zero_log_var(z_r),
            "mu_t": z_t, "log_var_t": zero_log_var(z_t),
            "loss/adv_pitch_on_zt": adv_pitch_on_zt,
            "loss/adv_rhythm_on_zt": adv_rhythm_on_zt,
            "loss/adv_pitch_on_zr": adv_pitch_on_zr,
            "loss/adv_envelope_on_zr": adv_envelope_on_zr,
            "loss/content_adv": content_adv_loss,
        }

            # ==================================================================
    # 增加给 Inference 用的极简解码器接口
    # ==================================================================
    # 增加给 Inference 用的极简解码器接口
    # ==================================================================
    @torch.no_grad()
    def decode(
        self,
        z_p: Tensor,  # 期待 shape: [B, K, n_bins, T] (4D)
        z_r: Tensor,  # 期待 shape: [B, d_r, T] (3D)
        z_t: Tensor,  # 期待 shape: [B, d_t] (2D)
        z_content: Optional[Tensor] = None
    ) -> Tensor:
        """
        专门给 DiT Inference 用的解码接口。
        """
        # 第一步：把 4D 的音高直接丢给原生 PitchEmbed (它自己会 flatten)
        pitch_emb = self.pitch_embed(z_p)
        
        # 第二步：把 p, r, t 融合
        h = self.fusion(pitch_emb, z_r, z_t)
        
        # 第三步：如果有残差特征 z_content，加上它
        if self.content_bottleneck is not None and z_content is not None:
            h = h + self.content_proj(z_content)
            
        # 第四步：丢给 SEANet 渲染出声音！
        x_hat = self.decoder(h)
        return x_hat
