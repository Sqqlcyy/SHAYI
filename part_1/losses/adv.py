from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm, spectral_norm


DiscOut = List[Tuple[Tensor, List[Tensor]]]


# ---------------------------------------------------------------------------
# MPD — Multi-Period Discriminator 
# ---------------------------------------------------------------------------
class _PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, kernel: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm
        ks = (kernel, 1)
        st = (stride, 1)
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1,   32, ks, st, padding=(kernel // 2, 0))),
            norm(nn.Conv2d(32, 128, ks, st, padding=(kernel // 2, 0))),
            norm(nn.Conv2d(128, 512, ks, st, padding=(kernel // 2, 0))),
            norm(nn.Conv2d(512, 1024, ks, st, padding=(kernel // 2, 0))),
            norm(nn.Conv2d(1024, 1024, (kernel, 1), 1, padding=(kernel // 2, 0))),
        ])
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        # x: [B, 1, T] -> reshape into [B, 1, T/P, P]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        pad = (self.period - (T % self.period)) % self.period
        if pad > 0:
            x = F.pad(x, (0, pad), mode="reflect")
            T = T + pad
        x = x.view(B, C, T // self.period, self.period)

        feats: List[Tensor] = []
        h = x
        for conv in self.convs:
            h = conv(h)
            h = F.leaky_relu(h, 0.1)
            feats.append(h)
        h = self.conv_post(h)
        feats.append(h)
        return h.flatten(1), feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [_PeriodDiscriminator(p) for p in periods]
        )

    def forward(self, x: Tensor) -> DiscOut:
        return [d(x) for d in self.discriminators]


# ---------------------------------------------------------------------------
# MSD — Multi-Scale Discriminator (MelGAN / HiFi-GAN)
# ---------------------------------------------------------------------------
class _ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,    128, 15, 1, padding=7)),
            norm(nn.Conv1d(128,  128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128,  256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256,  512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feats: List[Tensor] = []
        h = x
        for conv in self.convs:
            h = conv(h)
            h = F.leaky_relu(h, 0.1)
            feats.append(h)
        h = self.conv_post(h)
        feats.append(h)
        return h.flatten(1), feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales: int = 3):
        super().__init__()
        # First scale typically uses spectral norm in HiFi-GAN.
        self.discriminators = nn.ModuleList([
            _ScaleDiscriminator(use_spectral_norm=(i == 0))
            for i in range(n_scales)
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(n_scales - 1)
        ])

    def forward(self, x: Tensor) -> DiscOut:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        outs: DiscOut = []
        h = x
        for i, d in enumerate(self.discriminators):
            if i > 0:
                h = self.pools[i - 1](h)
            outs.append(d(h))
        return outs


# ---------------------------------------------------------------------------
# MSSD — Multi-Scale Spectrogram Discriminator 
# ---------------------------------------------------------------------------
class _SpecDiscriminator(nn.Module):
    """2-D conv discriminator over STFT magnitude.

    Catches spectral artifacts the time-domain MPD+MSD miss — especially
    important for music where sustained harmonic content shows up as
    structured stripes in the spectrogram.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, use_spectral_norm: bool = False):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1,   32, (3, 9), stride=(1, 1), padding=(1, 4))),
            norm(nn.Conv2d(32,  64, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(64, 128, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(128, 256, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1))),
        ])
        self.conv_post = norm(nn.Conv2d(256, 1, (3, 3), stride=(1, 1), padding=(1, 1)))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        if x.ndim == 3:
            x = x.squeeze(1)                            # [B, T]
        # STFT magnitude — keep dtype float32 for stability inside AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            spec = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(x.device),
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            mag = spec.abs().unsqueeze(1)               # [B, 1, F, T_spec]
        h = mag
        feats: List[Tensor] = []
        for conv in self.convs:
            h = conv(h)
            h = F.leaky_relu(h, 0.1)
            feats.append(h)
        h = self.conv_post(h)
        feats.append(h)
        return h.flatten(1), feats


class MultiScaleSpectrogramDiscriminator(nn.Module):
    """Three STFT resolutions (short / medium / long windows)."""

    _DEFAULT_CONFIGS: Tuple[Tuple[int, int, int], ...] = (
        (256,  64,  256),    # short window — captures transients (drums, onsets)
        (1024, 256, 1024),   # medium
        (4096, 1024, 4096),  # long — captures sustained harmonics
    )

    def __init__(self, configs: Optional[Tuple[Tuple[int, int, int], ...]] = None):
        super().__init__()
        configs = configs or self._DEFAULT_CONFIGS
        self.discriminators = nn.ModuleList([
            _SpecDiscriminator(n_fft=n, hop_length=h, win_length=w, use_spectral_norm=(i == 0))
            for i, (n, h, w) in enumerate(configs)
        ])

    def forward(self, x: Tensor) -> DiscOut:
        return [d(x) for d in self.discriminators]


# ---------------------------------------------------------------------------
# Combined wrapper
# ---------------------------------------------------------------------------
class CombinedDiscriminator(nn.Module):

    def __init__(
        self,
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        n_scales: int = 3,
        use_mssd: bool = False,
        mssd_configs: Optional[Tuple[Tuple[int, int, int], ...]] = None,
    ):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods=periods)
        self.msd = MultiScaleDiscriminator(n_scales=n_scales)
        self.mssd: Optional[MultiScaleSpectrogramDiscriminator] = (
            MultiScaleSpectrogramDiscriminator(mssd_configs) if use_mssd else None
        )

    def forward(self, x: Tensor) -> DiscOut:
        outs = self.mpd(x) + self.msd(x)
        if self.mssd is not None:
            outs = outs + self.mssd(x)
        return outs


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def hinge_d_loss(real_out: DiscOut, fake_out: DiscOut) -> Tensor:
    loss = real_out[0][0].new_zeros(())
    for (r_logits, _), (f_logits, _) in zip(real_out, fake_out):
        loss = loss + F.relu(1.0 - r_logits).mean() + F.relu(1.0 + f_logits).mean()
    return loss / max(len(real_out), 1)


def hinge_g_loss(fake_out: DiscOut) -> Tensor:
    loss = fake_out[0][0].new_zeros(())
    for f_logits, _ in fake_out:
        loss = loss - f_logits.mean()
    return loss / max(len(fake_out), 1)


def feature_matching_loss(real_out: DiscOut, fake_out: DiscOut) -> Tensor:
    loss = real_out[0][0].new_zeros(())
    n = 0
    for (_, r_feats), (_, f_feats) in zip(real_out, fake_out):
        for rf, ff in zip(r_feats, f_feats):
            loss = loss + F.l1_loss(ff, rf.detach())
            n += 1
    return loss / max(n, 1)
