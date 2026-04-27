"""Visualization helpers for AE reconstruction evaluation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def _ensure_matplotlib():
    cache_root = Path(os.environ.get("MPLCONFIGDIR", "/tmp/shayi_matplotlib"))
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.05:
        y = y / peak
    sf.write(str(path), y, sample_rate)


def _mel_db(audio: np.ndarray, sample_rate: int, n_mels: int = 128) -> np.ndarray:
    import torchaudio.transforms as T

    y = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, -1))
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels,
        power=2.0,
    )(y).squeeze(0)
    return (10.0 * torch.log10(mel.clamp_min(1e-10))).cpu().numpy()


def plot_mel_pair(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    sample_rate: int,
    out_path: Path,
    title: str,
) -> None:
    plt = _ensure_matplotlib()
    ref_db = _mel_db(reference, sample_rate)
    rec_db = _mel_db(reconstruction, sample_rate)
    vmin = float(min(np.percentile(ref_db, 5), np.percentile(rec_db, 5)))
    vmax = float(max(np.percentile(ref_db, 99), np.percentile(rec_db, 99)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].imshow(ref_db, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    axes[0].set_title("Reference")
    axes[0].set_ylabel("mel bin")
    im = axes[1].imshow(rec_db, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstruction")
    axes[1].set_ylabel("mel bin")
    axes[1].set_xlabel("frame")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label="dB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_waveform_pair(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    sample_rate: int,
    out_path: Path,
    title: str,
    max_seconds: float = 10.0,
) -> None:
    plt = _ensure_matplotlib()
    n = min(len(reference), len(reconstruction), int(sample_rate * max_seconds))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    axes[0].plot(t, reference[:n], linewidth=0.5, color="black")
    axes[0].set_title("Reference")
    axes[0].set_ylabel("amp")
    axes[1].plot(t, reconstruction[:n], linewidth=0.5, color="tab:blue")
    axes[1].set_title("Reconstruction")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("amp")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(records: List[Dict], out_path: Path) -> None:
    if not records:
        return
    plt = _ensure_matplotlib()
    idx = np.arange(len(records))
    metrics = ["snr", "mse", "l1", "recon_total", "recon_log_mel"]
    present = [m for m in metrics if any(m in r for r in records)]
    if not present:
        return

    fig, axes = plt.subplots(len(present), 1, figsize=(9, 2.3 * len(present)), sharex=True)
    if len(present) == 1:
        axes = [axes]
    for ax, key in zip(axes, present):
        vals = [float(r.get(key, np.nan)) for r in records]
        ax.plot(idx, vals, "o-", markersize=3, linewidth=1.0)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("eval item")
    fig.suptitle("Reconstruction Metrics")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
