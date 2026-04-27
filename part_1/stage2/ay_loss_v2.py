from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from losses.scalar_equiv import (
    PESTOPowerSeriesLoss,
    QuintonRatioLoss,
    TimbreAdditiveLoss,
)
from transforms import (
    FEATURE_TRANSFORM_OPS,
    FeatureTransformConfig,
    FeatureTransformFamily,
    build_aux_bundle,
)


def _mse(a: Tensor, b: Tensor) -> Tensor:
    # Shape-tolerant MSE: align time axes via truncation if they differ.
    if a.ndim >= 3 and b.ndim >= 3 and a.shape[-1] != b.shape[-1]:
        T = min(a.shape[-1], b.shape[-1])
        a = a[..., :T]
        b = b[..., :T]
    return F.mse_loss(a, b)


@dataclass
class AYConfigV2:
    # Feature transform ranges (same as v1 — same augmentation pipeline).
    pitch_min_st: float = -4.0
    pitch_max_st: float = 4.0
    rhythm_min_scale: float = 0.9
    rhythm_max_scale: float = 1.1111
    timbre_min_tilt_db: float = -6.0
    timbre_max_tilt_db: float = 6.0

    transform_backend: str = "feature"
    aux_sync_policy: str = "closed_form"

    feature_transform_ops: Tuple[str, ...] = field(default_factory=lambda: FEATURE_TRANSFORM_OPS)
    pitch_bins_per_semitone: int = 7
    vibrato_min_amount: float = 0.5
    vibrato_max_amount: float = 2.0
    vibrato_lfo_hz: float = 5.0
    feature_fps: float = 86.1328125
    rhythm_shift_min_steps: int = -16
    rhythm_shift_max_steps: int = 16
    timbre_high_cut_min_hz: float = 3000.0
    timbre_high_cut_max_hz: float = 12000.0
    timbre_formant_min_scale: float = 0.85
    timbre_formant_max_scale: float = 1.15
    mel_fmin_hz: float = 0.0
    mel_fmax_hz: float = 22050.0

    # Loss weights
    w_equiv: float = 1.0
    w_inv: float = 1.0
    w_comm: float = 0.5

    # Scalar-projection loss hyperparameters
    pesto_n_pitch_bins: int = 128       # matches model's head_p.n_pitch_bins
    pesto_bins_per_semitone: int = 1    # head_p outputs a semitone-spaced grid
    scalar_tau: float = 1.0
    timbre_gamma_init: float = 1.0

    include_rhythm_equivariance: bool = True
    include_commutativity: bool = True


class AnalyticYieldingLossV2(nn.Module):

    def __init__(self, cfg: Optional[AYConfigV2] = None):
        super().__init__()
        cfg = cfg or AYConfigV2()
        if cfg.aux_sync_policy != "closed_form":
            raise ValueError(f"Unsupported aux_sync_policy={cfg.aux_sync_policy!r}")
        if cfg.transform_backend.lower() not in {"feature", "analytic_feature", "features"}:
            raise ValueError(
                f"Unsupported transform_backend={cfg.transform_backend}; "
                "Stage 2 v2 currently only supports feature-domain transforms."
            )
        self.cfg = cfg

        self.feature_family = FeatureTransformFamily(
            FeatureTransformConfig(
                enabled_ops=tuple(cfg.feature_transform_ops),
                pitch_min_st=cfg.pitch_min_st,
                pitch_max_st=cfg.pitch_max_st,
                pitch_bins_per_semitone=cfg.pitch_bins_per_semitone,
                vibrato_min_amount=cfg.vibrato_min_amount,
                vibrato_max_amount=cfg.vibrato_max_amount,
                vibrato_lfo_hz=cfg.vibrato_lfo_hz,
                feature_fps=cfg.feature_fps,
                rhythm_min_scale=cfg.rhythm_min_scale,
                rhythm_max_scale=cfg.rhythm_max_scale,
                rhythm_shift_min_steps=cfg.rhythm_shift_min_steps,
                rhythm_shift_max_steps=cfg.rhythm_shift_max_steps,
                timbre_min_tilt_db=cfg.timbre_min_tilt_db,
                timbre_max_tilt_db=cfg.timbre_max_tilt_db,
                timbre_high_cut_min_hz=cfg.timbre_high_cut_min_hz,
                timbre_high_cut_max_hz=cfg.timbre_high_cut_max_hz,
                timbre_formant_min_scale=cfg.timbre_formant_min_scale,
                timbre_formant_max_scale=cfg.timbre_formant_max_scale,
                mel_fmin_hz=cfg.mel_fmin_hz,
                mel_fmax_hz=cfg.mel_fmax_hz,
            )
        )

        # Scalar-projection equivariance losses.
        self.eq_pitch = PESTOPowerSeriesLoss(
            n_pitch_bins=cfg.pesto_n_pitch_bins,
            bins_per_semitone=cfg.pesto_bins_per_semitone,
            tau=cfg.scalar_tau,
        )
        self.eq_rhythm = QuintonRatioLoss(tau=cfg.scalar_tau)
        self.eq_timbre = TimbreAdditiveLoss(tau=cfg.scalar_tau, gamma_init=cfg.timbre_gamma_init)

    # ------------------------------------------------------------------
    @staticmethod
    def _encode_transformed(
        model,
        codec_latent,       # Tensor (Variant B) or None (Variant A)
        aux_bundle,
        *,
        feat_mel=None,      # Variant A: fixed mel input for content encoder
    ) -> Dict[str, Tensor]:
        if hasattr(model, "encode_from_codec_latent") and codec_latent is not None:
            # Variant B / B-SEANet: same codec latent + transformed aux
            return model.encode_from_codec_latent(
                codec_latent, *aux_bundle.encode_args()
            )
        # Variant A / A-Codec: encode-only with transformed features.
        # Uses encode_only() which skips the decoder entirely — saves
        # ~10GB activation memory and is semantically correct (L_AY only
        # needs z_p/z_r/z_t, not x_hat).
        aux_pitch, aux_rhythm, aux_timbre = aux_bundle.encode_args()
        # Guard: if any aux is None (empty bundle), can't encode
        if aux_pitch is None or aux_rhythm is None:
            raise RuntimeError(
                "_encode_transformed got None aux tensors. Ensure "
                "build_aux_bundle(is_variant_b=True) is used for L_AY."
            )
        env = aux_timbre if getattr(model, "uses_envelope_aux", False) else None
        if hasattr(model, "encode_only"):
            return model.encode_only(aux_pitch, aux_rhythm, feat_mel, env)
        return model(aux_pitch, aux_rhythm, feat_mel, env)

    def _pitch_equiv(
        self,
        z_p_anchor: Tensor,
        z_p_shifted: Tensor,
        n_semitones_tensor: Tensor,
    ) -> Tensor:
        return self.eq_pitch(z_p_anchor, z_p_shifted, n_semitones_tensor)

    def _rhythm_equiv(
        self,
        model,
        z_r_anchor: Tensor,
        z_r_shifted: Tensor,
        alpha_anchor: Tensor,
        alpha_shifted: Tensor,
    ) -> Tensor:
        """Quinton ratio loss using the model's rhythm_scalar_proj head."""
        z_i = model.rhythm_scalar_proj(z_r_anchor)
        z_j = model.rhythm_scalar_proj(z_r_shifted)
        return self.eq_rhythm(z_i, z_j, alpha_anchor, alpha_shifted)

    def _timbre_equiv(
        self,
        model,
        z_t_anchor: Tensor,
        z_t_shifted: Tensor,
        delta_db: Tensor,
    ) -> Tensor:
        """Additive brightness regression using the model's timbre_brightness_proj."""
        b_a = model.timbre_brightness_proj(z_t_anchor)
        b_s = model.timbre_brightness_proj(z_t_shifted)
        return self.eq_timbre(b_a, b_s, delta_db)

    def _invariance(
        self,
        factor: str,
        enc_t: Dict[str, Tensor],
        anchor: Dict[str, Tensor],
    ) -> Tensor:

        zp_a = anchor["z_p"].detach()
        zr_a = anchor["z_r"].detach()
        zt_a = anchor["z_t"].detach()

        if factor == "pitch":
            return _mse(enc_t["z_r"], zr_a) + _mse(enc_t["z_t"], zt_a)
        if factor == "rhythm":
            return _mse(enc_t["z_p"], zp_a) + _mse(enc_t["z_t"], zt_a)
        if factor == "timbre":
            return _mse(enc_t["z_p"], zp_a) + _mse(enc_t["z_r"], zr_a)
        raise ValueError(f"Unknown factor: {factor!r}")

    def forward(
        self,
        model,
        anchor: Dict[str, Tensor],
        x: Tensor,                              # noqa: ARG002 (kept for trainer API parity)
        batch: Dict,
        curriculum: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, Tensor]:
        if not hasattr(model, "rhythm_scalar_proj") or not hasattr(model, "timbre_brightness_proj"):
            raise TypeError(
                "AnalyticYieldingLossV2 requires model with rhythm_scalar_proj and "
                "timbre_brightness_proj attributes (see heads.py)."
            )
        is_feature_mode = getattr(model, "input_mode", "waveform") == "feature"

        c_inv, c_equiv, c_comm = curriculum
        device = x.device

        # Anchor codec latent (Variant B) or None (Variant A feature-mode)
        codec_latent = anchor.get("feat")
        if codec_latent is None and not is_feature_mode:
            raise RuntimeError("AY v2 needs anchor['feat'] (the codec encoder output)")

        # feat_mel stays fixed across transforms (Variant A's content encoder invariant)
        feat_mel = batch.get("feat_mel")
        if feat_mel is not None:
            feat_mel = feat_mel.to(device, non_blocking=True)

        # Always build a populated AuxBundle: even feature-mode models need
        # aux features for L_AY transforms (pitch_shift, rhythm_stretch, etc.
        # operate on aux_pitch / aux_rhythm / aux_timbre).
        aux_anchor = build_aux_bundle(
            batch,
            device=device,
            is_variant_b=True,       # force populated bundle for all variants
            aux_zero_mask=getattr(getattr(model, "cfg", None), "aux_zero_mask", ()),
        )

        zero = x.new_zeros(())
        totals = {
            "inv_p": zero.clone(),
            "inv_r": zero.clone(),
            "inv_t": zero.clone(),
            "eq_p": zero.clone(),
            "eq_r": zero.clone(),
            "eq_t": zero.clone(),
        }
        counts = {"pitch": 0, "rhythm": 0, "timbre": 0}

        alpha_anchor = torch.ones(x.shape[0], device=device)  # anchor stretch = 1.0

        # Track per-op semitone / alpha / db for equivariance losses
        ops = tuple(self.feature_family.enabled_ops)
        for op in ops:
            spec = self.feature_family.sample_spec(op, device=device)
            aux_t = self.feature_family.apply(aux_anchor, spec)
            enc_t = self._encode_transformed(
                model, codec_latent, aux_t, feat_mel=feat_mel,
            )

            # Invariance contribution 
            L_inv = self._invariance(spec.factor, enc_t, anchor)
            totals[f"inv_{spec.factor[0]}"] = totals[f"inv_{spec.factor[0]}"] + L_inv

            # Equivariance contribution
            if spec.op == "pitch_shift":
                n_sem = x.new_tensor([float(spec.value)] * x.shape[0])
                L_eq = self._pitch_equiv(anchor["z_p"], enc_t["z_p"], n_sem)
                totals["eq_p"] = totals["eq_p"] + L_eq
                counts["pitch"] += 1
            elif spec.op == "rhythm_stretch":
                alpha_s = x.new_tensor([float(spec.value)] * x.shape[0])
                L_eq = self._rhythm_equiv(model, anchor["z_r"], enc_t["z_r"], alpha_anchor, alpha_s)
                totals["eq_r"] = totals["eq_r"] + L_eq
                counts["rhythm"] += 1
            elif spec.op == "timbre_eq_tilt":
                delta_db = x.new_tensor([float(spec.value)] * x.shape[0])
                L_eq = self._timbre_equiv(model, anchor["z_t"], enc_t["z_t"], delta_db)
                totals["eq_t"] = totals["eq_t"] + L_eq
                counts["timbre"] += 1

        # Normalize invariance contributions by op-count (consistent with v1)
        n_ops = max(len(ops), 1)
        for k in ("inv_p", "inv_r", "inv_t"):
            totals[k] = totals[k] / n_ops
        # Normalize equivariance by per-factor counts
        totals["eq_p"] = totals["eq_p"] / max(counts["pitch"], 1)
        totals["eq_r"] = totals["eq_r"] / max(counts["rhythm"], 1)
        totals["eq_t"] = totals["eq_t"] / max(counts["timbre"], 1)

        L_inv = totals["inv_p"] + totals["inv_r"] + totals["inv_t"]
        L_eq = totals["eq_p"] + totals["eq_r"] + totals["eq_t"]

        # Commutativity (optional) — test that (T_a ∘ T_b)(x) ≈ (T_b ∘ T_a)(x)
        L_comm = zero.clone()
        if self.cfg.include_commutativity:
            reps = (
                self.feature_family.sample_spec("pitch_shift", device=device),
                self.feature_family.sample_spec("rhythm_stretch", device=device),
                self.feature_family.sample_spec("timbre_eq_tilt", device=device),
            )
            comm_terms = []
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    a, b = reps[i], reps[j]
                    aux_ab = self.feature_family.apply(
                        self.feature_family.apply(aux_anchor, a), b
                    )
                    aux_ba = self.feature_family.apply(
                        self.feature_family.apply(aux_anchor, b), a
                    )
                    enc_ab = self._encode_transformed(model, codec_latent, aux_ab, feat_mel=feat_mel)
                    enc_ba = self._encode_transformed(model, codec_latent, aux_ba, feat_mel=feat_mel)
                    term = (
                        _mse(enc_ab["z_p"], enc_ba["z_p"])
                        + _mse(enc_ab["z_r"], enc_ba["z_r"])
                        + _mse(enc_ab["z_t"], enc_ba["z_t"])
                    )
                    comm_terms.append(term)
            if comm_terms:
                L_comm = torch.stack(comm_terms).mean()

        total = (
            c_inv * self.cfg.w_inv * L_inv
            + c_equiv * self.cfg.w_equiv * L_eq
            + c_comm * self.cfg.w_comm * L_comm
        )

        return {
            "ay/total": total,
            "ay/inv": L_inv.detach(),
            "ay/equiv": L_eq.detach(),
            "ay/comm": L_comm.detach(),
            "ay/inv_p": totals["inv_p"].detach(),
            "ay/inv_r": totals["inv_r"].detach(),
            "ay/inv_t": totals["inv_t"].detach(),
            "ay/eq_p": totals["eq_p"].detach(),
            "ay/eq_r": totals["eq_r"].detach(),
            "ay/eq_t": totals["eq_t"].detach(),
            "ay/curriculum_inv": torch.tensor(c_inv, device=device),
            "ay/curriculum_equiv": torch.tensor(c_equiv, device=device),
            "ay/curriculum_comm": torch.tensor(c_comm, device=device),
        }
