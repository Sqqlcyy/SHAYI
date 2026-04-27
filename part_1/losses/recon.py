from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _stft_magnitude(x: Tensor, n_fft: int, hop: int, win_length: int, window: Tensor) -> Tensor:
    # x: [B, T] or [B, 1, T] -> [B, F, N]
    if x.ndim == 3:
        x = x.squeeze(1)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )
    return spec.abs()


def _spectral_convergence(y_mag: Tensor, y_hat_mag: Tensor) -> Tensor:
    num = torch.linalg.norm(y_mag - y_hat_mag, ord="fro", dim=(-2, -1))
    den = torch.linalg.norm(y_mag, ord="fro", dim=(-2, -1)).clamp_min(1e-8)
    return (num / den).mean()


def _log_magnitude_l1(y_mag: Tensor, y_hat_mag: Tensor) -> Tensor:
    return F.l1_loss(torch.log(y_hat_mag + 1e-7), torch.log(y_mag + 1e-7))


class SingleSTFTLoss(nn.Module):
    def __init__(self, n_fft: int, hop: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        y_mag = _stft_magnitude(y, self.n_fft, self.hop, self.win_length, self.window)
        y_hat_mag = _stft_magnitude(y_hat, self.n_fft, self.hop, self.win_length, self.window)
        sc = _spectral_convergence(y_mag, y_hat_mag)
        lm = _log_magnitude_l1(y_mag, y_hat_mag)
        return sc, lm


class MultiResolutionSTFTLoss(nn.Module):

    def __init__(
        self,
        fft_sizes: List[int] = (1024, 2048, 4096),
        hop_sizes: List[int] = (120, 240, 480),
        win_lengths: List[int] = (960, 1920, 3840),
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), (
            "fft_sizes, hop_sizes, win_lengths must have the same length"
        )
        self.losses = nn.ModuleList(
            [SingleSTFTLoss(nf, hs, wl) for nf, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)]
        )

    def forward(self, y_hat: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        sc_total = y_hat.new_zeros(())
        lm_total = y_hat.new_zeros(())
        for loss in self.losses:
            sc, lm = loss(y_hat, y)
            sc_total = sc_total + sc
            lm_total = lm_total + lm
        n = len(self.losses)
        return sc_total / n, lm_total / n


# ---------------------------------------------------------------------------
# Log-mel L1 (per-frame)
# ---------------------------------------------------------------------------
class LogMelL1Loss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: float = None,
    ):
        super().__init__()
        import torchaudio.transforms as T

        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if y.ndim == 3:
            y = y.squeeze(1)
            y_hat = y_hat.squeeze(1)
        y_mel = self.mel(y)
        y_hat_mel = self.mel(y_hat)
        y_log = torch.log(y_mel + 1e-7)
        y_hat_log = torch.log(y_hat_mel + 1e-7)
        return F.l1_loss(y_hat_log, y_log)


# ---------------------------------------------------------------------------
# Composite Stage 1 reconstruction loss
# ---------------------------------------------------------------------------
class Stage1ReconLoss(nn.Module):
    """
    ``L_recon = L_mrstft_sc + L_mrstft_logmag + w_logmel * L1(log-mel) + w_wav_l1 * L1(wav)``
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        fft_sizes: List[int] = (1024, 2048, 4096),
        hop_sizes: List[int] = (120, 240, 480),
        win_lengths: List[int] = (960, 1920, 3840),
        n_mels: int = 128,
        w_logmel: float = 0.5,
        w_wav_l1: float = 1.0,
    ):
        super().__init__()
        self.mrstft = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)
        self.logmel = LogMelL1Loss(
            sample_rate=sample_rate,
            n_fft=max(fft_sizes),
            hop_length=hop_sizes[len(hop_sizes) // 2],
            n_mels=n_mels,
        )
        self.w_logmel = float(w_logmel)
        self.w_wav_l1 = float(w_wav_l1)

    def forward(self, y_hat: Tensor, y: Tensor) -> dict:
        sc, lm = self.mrstft(y_hat, y)
        lmel = self.logmel(y_hat, y)

        # Waveform-domain L1 — align channel dims defensively.
        y_w = y.squeeze(1) if y.ndim == 3 else y
        y_hat_w = y_hat.squeeze(1) if y_hat.ndim == 3 else y_hat
        T = min(y_w.shape[-1], y_hat_w.shape[-1])
        wav_l1 = F.l1_loss(y_hat_w[..., :T], y_w[..., :T])

        total = sc + lm + self.w_logmel * lmel + self.w_wav_l1 * wav_l1
        return {
            "recon/total": total,
            "recon/sc": sc.detach(),
            "recon/log_mag": lm.detach(),
            "recon/log_mel": lmel.detach(),
            "recon/wav_l1": wav_l1.detach(),
        }
