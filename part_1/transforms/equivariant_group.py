from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .pair_sampling import AuxBundle, _dct_basis, _shift_pitch_bins


FEATURE_TRANSFORM_OPS: Tuple[str, ...] = (
    "pitch_shift",
    "pitch_vibrato",
    "rhythm_stretch",
    "rhythm_shift",
    "timbre_high_cut",
    "timbre_eq_tilt",
    "timbre_formant",
)
FEATURE_TRANSFORM_ALIASES = {
    # Name used by the uploaded single-sample music_equivariant_group.py.
    "timbre_high_cut_eq": "timbre_high_cut",
}


@dataclass(frozen=True)
class FeatureTransformSpec:
    op: str
    factor: str
    value: float
    units: str


@dataclass
class FeatureTransformConfig:
    enabled_ops: Tuple[str, ...] = field(default_factory=lambda: FEATURE_TRANSFORM_OPS)

    pitch_min_st: float = -4.0
    pitch_max_st: float = 4.0
    pitch_bins_per_semitone: int = 7
    vibrato_min_amount: float = 0.5
    vibrato_max_amount: float = 2.0
    vibrato_lfo_hz: float = 5.0
    feature_fps: float = 86.1328125  # 44100 / 512 by default

    rhythm_min_scale: float = 0.9
    rhythm_max_scale: float = 1.1111
    rhythm_shift_min_steps: int = -16
    rhythm_shift_max_steps: int = 16

    timbre_min_tilt_db: float = -6.0
    timbre_max_tilt_db: float = 6.0
    timbre_high_cut_min_hz: float = 3000.0
    timbre_high_cut_max_hz: float = 12000.0
    timbre_formant_min_scale: float = 0.85
    timbre_formant_max_scale: float = 1.15
    mel_fmin_hz: float = 0.0
    mel_fmax_hz: float = 22050.0


def _as_tuple(ops: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if ops is None:
        return FEATURE_TRANSFORM_OPS
    if isinstance(ops, str):
        ops = (ops,)
    return tuple(FEATURE_TRANSFORM_ALIASES.get(str(op), str(op)) for op in ops)


def _resize_time(aux: Optional[Tensor], scale: float, *, mode: str) -> Optional[Tensor]:
    if aux is None or aux.ndim != 3:
        return aux
    new_T = max(1, int(round(aux.shape[-1] * float(scale))))
    if new_T == aux.shape[-1]:
        return aux
    if mode == "nearest":
        return F.interpolate(aux, size=new_T, mode="nearest")
    return F.interpolate(aux, size=new_T, mode=mode, align_corners=False)


def _shift_time(aux: Optional[Tensor], steps: int) -> Optional[Tensor]:
    if aux is None or aux.ndim != 3 or steps == 0:
        return aux
    T = aux.shape[-1]
    if abs(steps) >= T:
        return torch.zeros_like(aux)
    out = torch.zeros_like(aux)
    if steps > 0:
        out[..., steps:] = aux[..., : T - steps]
    else:
        s = -steps
        out[..., : T - s] = aux[..., s:]
    return out


def _vibrato_pitch_bins(
    aux: Optional[Tensor],
    *,
    amount: float,
    lfo_hz: float,
    fps: float,
    pitch_bins_per_semitone: int,
) -> Optional[Tensor]:
    if aux is None or aux.ndim != 3 or abs(float(amount)) < 1e-8:
        return aux
    B, C, T = aux.shape
    if C <= 1 or T == 0:
        return aux
    t = torch.arange(T, device=aux.device, dtype=aux.dtype) / float(fps)
    semitone = float(amount) * torch.sin(2.0 * math.pi * float(lfo_hz) * t)
    shifts = torch.round(semitone * int(pitch_bins_per_semitone)).to(torch.long)
    if int(shifts.abs().max().item()) == 0:
        return aux

    # Positive shift moves energy to larger pitch-bin indices:
    # out[:, c_out, t] = aux[:, c_out - shift[t], t].
    bins = torch.arange(C, device=aux.device).view(1, C, 1)
    src = bins - shifts.view(1, 1, T)
    valid = (src >= 0) & (src < C)
    src = src.clamp(0, C - 1).expand(B, C, T)
    out = aux.gather(dim=1, index=src)
    return out * valid.to(dtype=aux.dtype)


def _power_to_db(mel_linear: np.ndarray) -> np.ndarray:
    mel_db = 10.0 * np.log10(np.maximum(mel_linear, 1e-10))
    return (mel_db - float(np.max(mel_db))).astype(np.float32)


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + max(float(hz), 0.0) / 700.0)


def _mel_bin_for_hz(cutoff_hz: float, n_mels: int, fmin: float, fmax: float) -> int:
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_cut = min(max(_hz_to_mel(cutoff_hz), mel_min), mel_max)
    ratio = (mel_cut - mel_min) / max(mel_max - mel_min, 1e-8)
    return int(round(ratio * (n_mels - 1)))


def _mfcc_from_mel_paths(
    spec_paths: Optional[Sequence[str]],
    mel_linear_paths: Optional[Sequence[str]],
    base_timbre: Optional[Tensor],
    transform,
    feat_starts: Optional[Sequence[int]] = None,
    feat_lengths: Optional[Sequence[int]] = None,
) -> Optional[Tensor]:
    if base_timbre is None:
        return base_timbre

    base_np = base_timbre.detach().cpu().numpy().astype(np.float32)
    out = base_np.copy()
    is_per_frame = base_np.ndim == 3
    spec_paths = spec_paths or ()
    mel_linear_paths = mel_linear_paths or ()
    for i in range(base_np.shape[0]):
        linear_path = mel_linear_paths[i] if i < len(mel_linear_paths) else ""
        spec_path = spec_paths[i] if i < len(spec_paths) else ""
        if linear_path:
            mel = np.load(Path(linear_path)).astype(np.float32)
            if mel.ndim != 2:
                raise ValueError(f"Expected linear mel [n_mels, T] at {linear_path}, got {mel.shape}")
        elif spec_path:
            spec = np.load(Path(spec_path)).astype(np.float32)
            if spec.ndim != 2:
                raise ValueError(f"Expected mel-dB spec [n_mels, T] at {spec_path}, got {spec.shape}")
            mel = np.power(10.0, spec / 10.0).astype(np.float32)
        else:
            out[i] = base_np[i]
            continue
        mel_t = transform(mel)
        mel_db_t = _power_to_db(mel_t)
        basis = _dct_basis(int(base_np.shape[1]), int(mel_db_t.shape[0]))
        mfcc = basis @ mel_db_t
        if is_per_frame:
            start = int(feat_starts[i]) if feat_starts is not None and i < len(feat_starts) else 0
            length = int(feat_lengths[i]) if feat_lengths is not None and i < len(feat_lengths) else int(base_np.shape[2])
            crop = mfcc[:, start : min(start + length, mfcc.shape[1])]
            if crop.shape[1] == 0:
                crop = np.zeros((base_np.shape[1], 1), dtype=np.float32)
            x = torch.from_numpy(crop.astype(np.float32))[None, :, :]
            y = F.interpolate(x, size=int(base_np.shape[2]), mode="linear", align_corners=False)
            out[i] = y.squeeze(0).numpy().astype(np.float32)
        else:
            out[i] = mfcc.mean(axis=1).astype(np.float32)
    return torch.from_numpy(out).to(device=base_timbre.device, dtype=base_timbre.dtype)


def _apply_eq_tilt_linear(mel: np.ndarray, tilt_db: float) -> np.ndarray:
    gains_db = np.linspace(-tilt_db / 2.0, tilt_db / 2.0, num=mel.shape[0], dtype=np.float32)[:, None]
    gains_linear = np.power(10.0, gains_db / 20.0).astype(np.float32)
    return mel * gains_linear


def _apply_high_cut_linear(mel: np.ndarray, cutoff_hz: float, *, mel_fmin_hz: float, mel_fmax_hz: float) -> np.ndarray:
    n_mels = mel.shape[0]
    cut_start = _mel_bin_for_hz(cutoff_hz, n_mels, mel_fmin_hz, mel_fmax_hz)
    if cut_start >= n_mels - 1:
        return mel.copy()
    cut_start = max(0, cut_start)
    out = mel.copy()
    decay_len = n_mels - cut_start
    t = np.linspace(0.0, np.pi / 2.0, num=decay_len, dtype=np.float32)
    linear_gain = 0.1 + 0.9 * np.cos(t)
    out[cut_start:, :] = out[cut_start:, :] * linear_gain[:, None]
    return out


def _apply_formant_warp_linear(mel: np.ndarray, formant_scale: float) -> np.ndarray:
    if mel.shape[0] < 2 or mel.shape[1] < 2 or abs(float(formant_scale) - 1.0) < 1e-8:
        return mel.copy()
    n_mels, T = mel.shape
    new_mels = max(2, int(round(n_mels * float(formant_scale))))
    x = torch.from_numpy(mel).view(1, 1, n_mels, T)
    scaled = F.interpolate(x, size=(new_mels, T), mode="bilinear", align_corners=False)
    scaled_np = scaled.view(new_mels, T).numpy().astype(np.float32)
    if new_mels >= n_mels:
        out = scaled_np[:n_mels, :]
    else:
        out = np.zeros_like(mel)
        out[:new_mels, :] = scaled_np
        tail_value = scaled_np[-1:, :]
        remaining = n_mels - new_mels
        if remaining > 0:
            decay = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, num=remaining, dtype=np.float32)))
            out[new_mels:, :] = tail_value * decay[:, None]
    return out


class FeatureTransformFamily:
    """Seven feature-domain transforms used by the mainline L_AY path."""

    def __init__(self, cfg: Optional[FeatureTransformConfig] = None):
        self.cfg = cfg or FeatureTransformConfig()
        self.enabled_ops = _as_tuple(self.cfg.enabled_ops)
        unknown = sorted(set(self.enabled_ops) - set(FEATURE_TRANSFORM_OPS))
        if unknown:
            raise ValueError(f"Unknown feature transform op(s): {unknown}")

    @staticmethod
    def op_factor(op: str) -> str:
        if op.startswith("pitch_"):
            return "pitch"
        if op.startswith("rhythm_"):
            return "rhythm"
        if op.startswith("timbre_"):
            return "timbre"
        raise ValueError(f"Unknown feature transform op: {op}")

    def sample_spec(self, op: Optional[str] = None, *, device: Optional[torch.device] = None) -> FeatureTransformSpec:
        if op is None:
            idx = int(torch.randint(0, len(self.enabled_ops), (1,), device=device).item())
            op = self.enabled_ops[idx]
        op = str(op)
        cfg = self.cfg

        if op == "pitch_shift":
            lo = math.ceil(cfg.pitch_min_st)
            hi = math.floor(cfg.pitch_max_st)
            value = int(torch.randint(lo, hi + 1, (1,), device=device).item()) if lo <= hi else int(round(cfg.pitch_min_st))
            if value == 0 and lo < hi:
                value = hi if abs(hi) >= abs(lo) else lo
            return FeatureTransformSpec(op=op, factor="pitch", value=float(value), units="semitones")
        if op == "pitch_vibrato":
            value = float(torch.empty(1, device=device).uniform_(cfg.vibrato_min_amount, cfg.vibrato_max_amount).item())
            return FeatureTransformSpec(op=op, factor="pitch", value=value, units="semitone_amplitude")
        if op == "rhythm_stretch":
            value = float(torch.empty(1, device=device).uniform_(cfg.rhythm_min_scale, cfg.rhythm_max_scale).item())
            return FeatureTransformSpec(op=op, factor="rhythm", value=value, units="scale")
        if op == "rhythm_shift":
            lo = int(cfg.rhythm_shift_min_steps)
            hi = int(cfg.rhythm_shift_max_steps)
            value = int(torch.randint(lo, hi + 1, (1,), device=device).item())
            if value == 0 and lo < hi:
                value = hi if abs(hi) >= abs(lo) else lo
            return FeatureTransformSpec(op=op, factor="rhythm", value=float(value), units="feature_steps")
        if op == "timbre_high_cut":
            value = float(torch.empty(1, device=device).uniform_(cfg.timbre_high_cut_min_hz, cfg.timbre_high_cut_max_hz).item())
            return FeatureTransformSpec(op=op, factor="timbre", value=value, units="Hz")
        if op == "timbre_eq_tilt":
            value = float(torch.empty(1, device=device).uniform_(cfg.timbre_min_tilt_db, cfg.timbre_max_tilt_db).item())
            return FeatureTransformSpec(op=op, factor="timbre", value=value, units="dB")
        if op == "timbre_formant":
            value = float(torch.empty(1, device=device).uniform_(cfg.timbre_formant_min_scale, cfg.timbre_formant_max_scale).item())
            return FeatureTransformSpec(op=op, factor="timbre", value=value, units="scale")
        raise ValueError(f"Unknown feature transform op: {op}")

    def action_strength(self, spec: FeatureTransformSpec) -> float:
        cfg = self.cfg
        if spec.op == "pitch_shift":
            denom = max(abs(cfg.pitch_min_st), abs(cfg.pitch_max_st), 1e-6)
            return min(abs(spec.value) / denom, 1.0)
        if spec.op == "pitch_vibrato":
            denom = max(abs(cfg.vibrato_max_amount), 1e-6)
            return min(abs(spec.value) / denom, 1.0)
        if spec.op == "rhythm_stretch":
            denom = max(abs(math.log(cfg.rhythm_min_scale)), abs(math.log(cfg.rhythm_max_scale)), 1e-6)
            return min(abs(math.log(max(spec.value, 1e-6))) / denom, 1.0)
        if spec.op == "rhythm_shift":
            denom = max(abs(cfg.rhythm_shift_min_steps), abs(cfg.rhythm_shift_max_steps), 1)
            return min(abs(spec.value) / denom, 1.0)
        if spec.op == "timbre_high_cut":
            span = max(cfg.timbre_high_cut_max_hz - cfg.timbre_high_cut_min_hz, 1e-6)
            return min(max((cfg.timbre_high_cut_max_hz - spec.value) / span, 0.0), 1.0)
        if spec.op == "timbre_eq_tilt":
            denom = max(abs(cfg.timbre_min_tilt_db), abs(cfg.timbre_max_tilt_db), 1e-6)
            return min(abs(spec.value) / denom, 1.0)
        if spec.op == "timbre_formant":
            denom = max(abs(math.log(cfg.timbre_formant_min_scale)), abs(math.log(cfg.timbre_formant_max_scale)), 1e-6)
            return min(abs(math.log(max(spec.value, 1e-6))) / denom, 1.0)
        return 0.0

    def apply(self, bundle: AuxBundle, spec: FeatureTransformSpec) -> AuxBundle:
        cfg = self.cfg
        if spec.op == "pitch_shift":
            bin_shift = int(round(spec.value)) * int(cfg.pitch_bins_per_semitone)
            return AuxBundle(
                pitch=_shift_pitch_bins(bundle.pitch, bin_shift),
                rhythm=bundle.rhythm,
                timbre=bundle.timbre,
                spec_paths=bundle.spec_paths,
                mel_linear_paths=bundle.mel_linear_paths,
                feat_starts=bundle.feat_starts,
                feat_lengths=bundle.feat_lengths,
            )
        if spec.op == "pitch_vibrato":
            return AuxBundle(
                pitch=_vibrato_pitch_bins(
                    bundle.pitch,
                    amount=float(spec.value),
                    lfo_hz=cfg.vibrato_lfo_hz,
                    fps=cfg.feature_fps,
                    pitch_bins_per_semitone=cfg.pitch_bins_per_semitone,
                ),
                rhythm=bundle.rhythm,
                timbre=bundle.timbre,
                spec_paths=bundle.spec_paths,
                mel_linear_paths=bundle.mel_linear_paths,
                feat_starts=bundle.feat_starts,
                feat_lengths=bundle.feat_lengths,
            )
        if spec.op == "rhythm_stretch":
            return AuxBundle(
                pitch=_resize_time(bundle.pitch, float(spec.value), mode="linear"),
                rhythm=_resize_time(bundle.rhythm, float(spec.value), mode="nearest"),
                timbre=bundle.timbre,
                spec_paths=bundle.spec_paths,
                mel_linear_paths=bundle.mel_linear_paths,
                feat_starts=bundle.feat_starts,
                feat_lengths=bundle.feat_lengths,
            )
        if spec.op == "rhythm_shift":
            steps = int(round(float(spec.value)))
            return AuxBundle(
                pitch=_shift_time(bundle.pitch, steps),
                rhythm=_shift_time(bundle.rhythm, steps),
                timbre=bundle.timbre,
                spec_paths=bundle.spec_paths,
                mel_linear_paths=bundle.mel_linear_paths,
                feat_starts=bundle.feat_starts,
                feat_lengths=bundle.feat_lengths,
            )
        if spec.op == "timbre_high_cut":
            timbre = _mfcc_from_mel_paths(
                bundle.spec_paths,
                bundle.mel_linear_paths,
                bundle.timbre,
                lambda mel: _apply_high_cut_linear(
                    mel,
                    float(spec.value),
                    mel_fmin_hz=cfg.mel_fmin_hz,
                    mel_fmax_hz=cfg.mel_fmax_hz,
                ),
                bundle.feat_starts,
                bundle.feat_lengths,
            )
            return AuxBundle(
                pitch=bundle.pitch,
                rhythm=bundle.rhythm,
                timbre=timbre,
                spec_paths=bundle.spec_paths,
                mel_linear_paths=bundle.mel_linear_paths,
                feat_starts=bundle.feat_starts,
                feat_lengths=bundle.feat_lengths,
            )
        if spec.op == "timbre_eq_tilt":
            timbre = _mfcc_from_mel_paths(
                bundle.spec_paths,
                bundle.mel_linear_paths,
                bundle.timbre,
                lambda mel: _apply_eq_tilt_linear(mel, float(spec.value)),
                bundle.feat_starts,
                bundle.feat_lengths,
            )
            return AuxBundle(
                pitch=bundle.pitch,
                rhythm=bundle.rhythm,
                timbre=timbre,
                spec_paths=bundle.spec_paths,
                mel_linear_paths=bundle.mel_linear_paths,
                feat_starts=bundle.feat_starts,
                feat_lengths=bundle.feat_lengths,
            )
        if spec.op == "timbre_formant":
            timbre = _mfcc_from_mel_paths(
                bundle.spec_paths,
                bundle.mel_linear_paths,
                bundle.timbre,
                lambda mel: _apply_formant_warp_linear(mel, float(spec.value)),
                bundle.feat_starts,
                bundle.feat_lengths,
            )
            return AuxBundle(
                pitch=bundle.pitch,
                rhythm=bundle.rhythm,
                timbre=timbre,
                spec_paths=bundle.spec_paths,
                mel_linear_paths=bundle.mel_linear_paths,
                feat_starts=bundle.feat_starts,
                feat_lengths=bundle.feat_lengths,
            )
        raise ValueError(f"Unknown feature transform op: {spec.op}")
