from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor


def load_mono_audio(path: Union[str, Path], target_sr: int = None) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if target_sr is None or int(sr) == int(target_sr):
        return y.astype(np.float32), int(sr)

    try:
        import torch
        import torchaudio.functional as AF

        y_t = torch.from_numpy(y.astype(np.float32))[None, :]
        y_rs = AF.resample(y_t, orig_freq=int(sr), new_freq=int(target_sr)).squeeze(0)
        return y_rs.cpu().numpy().astype(np.float32), int(target_sr)
    except Exception:  # noqa: BLE001
        import librosa

        y_rs = librosa.resample(y.astype(np.float32), orig_sr=int(sr), target_sr=int(target_sr))
        return y_rs.astype(np.float32), int(target_sr)


def peak_normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < eps:
        return y.astype(np.float32)
    return (y / peak).astype(np.float32)


def random_crop_1d(y: np.ndarray, length: int, rng: np.random.Generator = None) -> np.ndarray:
    """Random fixed-length crop of a 1-D waveform. Zero-pads if too short."""
    n = y.shape[0]
    if n == length:
        return y.astype(np.float32)
    if n < length:
        out = np.zeros(length, dtype=np.float32)
        out[:n] = y
        return out
    if rng is None:
        start = int(np.random.randint(0, n - length + 1))
    else:
        start = int(rng.integers(0, n - length + 1))
    return y[start : start + length].astype(np.float32)


def center_crop_1d(y: np.ndarray, length: int) -> np.ndarray:
    n = y.shape[0]
    if n == length:
        return y.astype(np.float32)
    if n < length:
        out = np.zeros(length, dtype=np.float32)
        pad = (length - n) // 2
        out[pad : pad + n] = y
        return out
    start = (n - length) // 2
    return y[start : start + length].astype(np.float32)


def pad_or_crop_1d(y: Union[np.ndarray, Tensor], length: int) -> Union[np.ndarray, Tensor]:
    """Right-pad with zeros or right-truncate a 1-D array/tensor to ``length``."""
    if isinstance(y, Tensor):
        n = y.shape[-1]
        if n == length:
            return y
        if n < length:
            pad = length - n
            return torch.nn.functional.pad(y, (0, pad))
        return y[..., :length]
    else:
        n = y.shape[-1]
        if n == length:
            return y
        if n < length:
            out = np.zeros((*y.shape[:-1], length), dtype=y.dtype)
            out[..., :n] = y
            return out
        return y[..., :length]
