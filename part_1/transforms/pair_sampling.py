"""
Auxiliary-feature bundle helpers for Stage 1/2/3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class AuxBundle:
    pitch: Optional[Tensor] = None
    rhythm: Optional[Tensor] = None
    timbre: Optional[Tensor] = None
    spec_paths: Optional[Tuple[str, ...]] = None
    mel_linear_paths: Optional[Tuple[str, ...]] = None
    feat_starts: Optional[Tuple[int, ...]] = None
    feat_lengths: Optional[Tuple[int, ...]] = None

    def encode_args(self) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        return self.pitch, self.rhythm, self.timbre


def _mask_to_set(aux_zero_mask: Optional[Iterable[str]]) -> set[str]:
    if aux_zero_mask is None:
        return set()
    if isinstance(aux_zero_mask, str):
        return {aux_zero_mask}
    return {str(x) for x in aux_zero_mask}


def build_aux_bundle(
    batch: Dict,
    *,
    device: torch.device,
    is_variant_b: bool,
    aux_zero_mask: Optional[Iterable[str]] = None,
) -> AuxBundle:
    if not is_variant_b:
        return AuxBundle()

    pitch = batch["aux_pitch"].to(device, non_blocking=True)
    rhythm = batch["aux_rhythm"].to(device, non_blocking=True)
    timbre = batch["aux_timbre"].to(device, non_blocking=True)
    mask = _mask_to_set(aux_zero_mask)
    if "pitch" in mask:
        pitch = torch.zeros_like(pitch)
    if "rhythm" in mask:
        rhythm = torch.zeros_like(rhythm)
    if "timbre" in mask:
        timbre = torch.zeros_like(timbre)
    spec_paths = tuple(batch.get("spec_path", []))
    mel_linear_paths = tuple(batch.get("mel_linear_path", []))
    feat_starts_raw = batch.get("feat_start")
    feat_lengths_raw = batch.get("feat_length")
    feat_starts = None
    feat_lengths = None
    if isinstance(feat_starts_raw, torch.Tensor):
        feat_starts = tuple(int(v.item()) for v in feat_starts_raw.cpu())
    elif feat_starts_raw is not None:
        feat_starts = tuple(int(v) for v in feat_starts_raw)
    if isinstance(feat_lengths_raw, torch.Tensor):
        feat_lengths = tuple(int(v.item()) for v in feat_lengths_raw.cpu())
    elif feat_lengths_raw is not None:
        feat_lengths = tuple(int(v) for v in feat_lengths_raw)
    return AuxBundle(
        pitch=pitch,
        rhythm=rhythm,
        timbre=timbre,
        spec_paths=spec_paths,
        mel_linear_paths=mel_linear_paths,
        feat_starts=feat_starts,
        feat_lengths=feat_lengths,
    )


def _shift_pitch_bins(aux: Optional[Tensor], bin_shift: int) -> Optional[Tensor]:
    
    if aux is None or aux.ndim != 3 or bin_shift == 0:
        return aux
    n_bins = aux.shape[1]
    if n_bins <= 1:
        return aux
    if abs(bin_shift) >= n_bins:
        return torch.zeros_like(aux)
    out = torch.zeros_like(aux)
    if bin_shift > 0:
        out[:, bin_shift:, :] = aux[:, : n_bins - bin_shift, :]
    else:
        s = -bin_shift
        out[:, : n_bins - s, :] = aux[:, s:, :]
    return out


@lru_cache(maxsize=None)
def _dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    basis = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    basis[0, :] = 1.0 / math.sqrt(n_mels)
    if n_mfcc == 1:
        return basis
    scale = math.sqrt(2.0 / n_mels)
    n = np.arange(n_mels, dtype=np.float32)
    for k in range(1, n_mfcc):
        basis[k, :] = scale * np.cos((math.pi / n_mels) * (n + 0.5) * k)
    return basis
