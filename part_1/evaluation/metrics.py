"""Metrics shared by reconstruction inference and dashboards."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from losses.recon import Stage1ReconLoss
from runtime import align_audio_pair


def waveform_metrics_np(reference: np.ndarray, reconstruction: np.ndarray) -> Dict[str, float]:
    """Legacy-compatible waveform metrics: SNR, MSE, and L1.
    """
    n = min(int(reference.shape[-1]), int(reconstruction.shape[-1]))
    if n <= 0:
        return {"snr": float("nan"), "mse": float("nan"), "l1": float("nan"), "peak_abs_err": float("nan")}
    ref = reference[..., :n].reshape(-1).astype(np.float64)
    rec = reconstruction[..., :n].reshape(-1).astype(np.float64)
    err = ref - rec
    signal_power = float(np.mean(ref ** 2))
    noise_power = float(np.mean(err ** 2))
    if noise_power < 1e-12:
        snr = 100.0
    elif signal_power < 1e-12:
        snr = -100.0
    else:
        snr = 10.0 * math.log10(signal_power / noise_power)
    return {
        "snr": float(snr),
        "mse": float(noise_power),
        "l1": float(np.mean(np.abs(err))),
        "peak_abs_err": float(np.max(np.abs(err))),
    }


# ---------------------------------------------------------------------------
# Phase-invariant per-sample metrics (use these alongside SNR for a fair read)
# ---------------------------------------------------------------------------
def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR (Le Roux et al., 2019). Phase-sensitive still.

    Use in addition to SNR to isolate "energy ratio correct" from "scale correct".
    """
    ref = reference.astype(np.float64) - reference.mean()
    est = estimate.astype(np.float64) - estimate.mean()
    alpha = (ref @ est) / (ref @ ref + 1e-12)
    s_target = alpha * ref
    noise = est - s_target
    return 10.0 * math.log10((s_target @ s_target + 1e-12) / (noise @ noise + 1e-12))


def log_stft_l1(
    reference: np.ndarray,
    estimate: np.ndarray,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
) -> float:
    """Phase-invariant: L1 on log-STFT magnitude.

    Librosa-based (CPU, no torch). Good proxy for "does the energy land in the
    right frequency bins" without caring about waveform phase.
    """
    import librosa
    hop = hop_length or n_fft // 4
    R = np.abs(librosa.stft(reference.astype(np.float32), n_fft=n_fft, hop_length=hop))
    E = np.abs(librosa.stft(estimate.astype(np.float32), n_fft=n_fft, hop_length=hop))
    return float(np.mean(np.abs(np.log1p(R) - np.log1p(E))))


def mel_db_l1(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 44100,
    n_mels: int = 128,
) -> float:
    """Phase-invariant: L1 on log-mel (dB) — perceptually-weighted."""
    import librosa
    R = librosa.feature.melspectrogram(y=reference.astype(np.float32), sr=sample_rate, n_mels=n_mels)
    E = librosa.feature.melspectrogram(y=estimate.astype(np.float32), sr=sample_rate, n_mels=n_mels)
    return float(np.mean(np.abs(librosa.power_to_db(R) - librosa.power_to_db(E))))


def per_sample_recon_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 44100,
) -> Dict[str, float]:
    """Full per-sample recon bundle: SNR, SI-SDR, log-STFT L1, mel-dB L1."""
    n = min(len(reference), len(estimate))
    ref, est = reference[:n].astype(np.float32), estimate[:n].astype(np.float32)
    snr_info = waveform_metrics_np(ref, est)
    return {
        "snr_db": snr_info["snr"],
        "si_sdr": si_sdr(ref, est),
        "log_stft_l1": log_stft_l1(ref, est),
        "mel_db_l1": mel_db_l1(ref, est, sample_rate=sample_rate),
    }


class ReconstructionMetricComputer:
    """MR-STFT/log-mel metric wrapper using the training reconstruction loss."""

    def __init__(
        self,
        *,
        sample_rate: int = 32000,
        fft_sizes: Iterable[int] = (1024, 2048, 4096),
        hop_sizes: Iterable[int] = (120, 240, 480),
        win_lengths: Iterable[int] = (960, 1920, 3840),
        n_mels: int = 128,
        w_logmel: float = 0.5,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.loss = Stage1ReconLoss(
            sample_rate=sample_rate,
            fft_sizes=list(fft_sizes),
            hop_sizes=list(hop_sizes),
            win_lengths=list(win_lengths),
            n_mels=n_mels,
            w_logmel=w_logmel,
        ).to(self.device)

    @torch.no_grad()
    def compute(self, reconstruction: Tensor, reference: Tensor) -> Dict[str, float]:
        reconstruction, reference = align_audio_pair(reconstruction, reference)
        losses = self.loss(reconstruction.to(self.device), reference.to(self.device))
        return {
            "recon_total": float(losses["recon/total"].item()),
            "recon_sc": float(losses["recon/sc"].item()),
            "recon_log_mag": float(losses["recon/log_mag"].item()),
            "recon_log_mel": float(losses["recon/log_mel"].item()),
        }


class FADComputer:

    _SUPPORTED = {"vggish", "pann", "clap"}

    def __init__(
        self,
        model_name: str = "vggish",
        sample_rate: int = 16000,
        device: Union[torch.device, str] = "cpu",
    ):
        if model_name not in self._SUPPORTED:
            raise ValueError(f"FAD backbone {model_name!r} unsupported; pick {self._SUPPORTED}")
        self.model_name = model_name
        self.sample_rate = int(sample_rate)
        self.device = torch.device(device)
        self._fad = None

    def _lazy_init(self):
        if self._fad is not None:
            return self._fad
        try:
            from frechet_audio_distance import FrechetAudioDistance
        except ImportError as e:  # noqa: BLE001
            raise ImportError(
                "FAD requires `pip install frechet-audio-distance`."
            ) from e
        self._fad = FrechetAudioDistance(
            model_name=self.model_name,
            sample_rate=self.sample_rate,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )
        return self._fad

    def compute(
        self,
        reference_dir: Union[str, Path],
        reconstruction_dir: Union[str, Path],
    ) -> Dict[str, float]:
        fad = self._lazy_init()
        score = fad.score(str(reference_dir), str(reconstruction_dir))
        return {f"fad_{self.model_name}": float(score)}


def compute_fad_multi(
    reference_dir: Union[str, Path],
    reconstruction_dir: Union[str, Path],
    backbones: Iterable[str] = ("vggish",),
    device: Union[torch.device, str] = "cpu",
) -> Dict[str, float]:
    """Run FAD across multiple backbones and merge results."""
    out: Dict[str, float] = {}
    for name in backbones:
        computer = FADComputer(model_name=name, device=device)
        out.update(computer.compute(reference_dir, reconstruction_dir))
    return out


def summarize_records(records: List[Dict]) -> Dict[str, float | int]:
    """Mean/std summary for numeric metric fields."""
    out: Dict[str, float | int] = {"num_items": len(records)}
    if not records:
        return out
    keys = sorted(
        k for k, v in records[0].items()
        if isinstance(v, (int, float)) and k not in {"index", "batch_index"}
    )
    for key in keys:
        vals = np.array(
            [float(r[key]) for r in records if isinstance(r.get(key), (int, float)) and np.isfinite(float(r[key]))],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue
        out[f"{key}/mean"] = float(vals.mean())
        out[f"{key}/std"] = float(vals.std())
    return out
