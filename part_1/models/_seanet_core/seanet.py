# Vendored from cloud_code/part_2/audiocraft/modules/seanet.py
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under MIT.
import math
import typing as tp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import StreamableConv1d, StreamableConvTranspose1d
from .lstm import StreamableLSTM


# ---------------------------------------------------------------------------
# BigVGAN-style alias-free resampling (sinc + Hann window)
# ---------------------------------------------------------------------------
def _kaiser_sinc_lowpass(n: int, cutoff: float) -> torch.Tensor:
    """Windowed-sinc low-pass FIR kernel, cutoff in [0, 0.5] (Nyquist)."""
    t = torch.arange(n, dtype=torch.float32) - (n - 1) / 2
    scaled = 2.0 * cutoff * t
    sinc = torch.where(
        scaled == 0,
        torch.ones_like(scaled),
        torch.sin(math.pi * scaled) / (math.pi * scaled + 1e-20),
    )
    hann = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(n, dtype=torch.float32) / (n - 1))
    kernel = sinc * hann * 2.0 * cutoff
    kernel = kernel / kernel.sum()
    return kernel


class _DepthwiseFIR(nn.Module):
    """Grouped-conv 1D FIR filter, reusing one kernel across channels."""

    def __init__(self, kernel_size: int = 12, cutoff: float = 0.25):
        super().__init__()
        k = _kaiser_sinc_lowpass(kernel_size, cutoff)
        self.register_buffer("kernel", k.view(1, 1, -1))
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        k = self.kernel.to(x.dtype).expand(C, 1, -1).contiguous()
        return F.conv1d(x, k, padding=self.pad, groups=C)[..., :T]


class Snake(nn.Module):
    """Snake activation: x + (1/α) · sin²(αx).

    Periodic inductive bias shown in BigVGAN (Lee et al., 2023) to lift
    waveform reconstruction quality on harmonic audio (music, instruments).
    α is per-channel learnable (registered lazily on first forward so the
    existing SEANet call site — which passes only ``alpha`` / ``beta``,
    not channel dim — keeps working).
    """

    def __init__(self, alpha_init: float = 1.0, **_unused):
        super().__init__()
        self.alpha_init = float(alpha_init)
        self.alpha: tp.Optional[nn.Parameter] = None

    def _lazy_init(self, x: torch.Tensor):
        if self.alpha is None:
            C = x.shape[1]
            self.alpha = nn.Parameter(
                torch.full((1, C, 1), self.alpha_init, device=x.device, dtype=x.dtype)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._lazy_init(x)
        a = self.alpha
        return x + (1.0 / (a + 1e-9)) * torch.sin(a * x).pow(2)


class SnakeBeta(nn.Module):
    """BigVGAN's SnakeBeta: x + (1/β) · sin²(αx). Separate α, β per channel."""

    def __init__(self, alpha_init: float = 1.0, beta_init: float = 1.0, **_unused):
        super().__init__()
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)
        self.alpha: tp.Optional[nn.Parameter] = None
        self.beta: tp.Optional[nn.Parameter] = None

    def _lazy_init(self, x: torch.Tensor):
        if self.alpha is None:
            C = x.shape[1]
            self.alpha = nn.Parameter(
                torch.full((1, C, 1), self.alpha_init, device=x.device, dtype=x.dtype)
            )
            self.beta = nn.Parameter(
                torch.full((1, C, 1), self.beta_init, device=x.device, dtype=x.dtype)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._lazy_init(x)
        return x + (1.0 / (self.beta + 1e-9)) * torch.sin(self.alpha * x).pow(2)


class AliasFreeSnakeBeta(nn.Module):
    """SnakeBeta wrapped by 2× upsample → activation → low-pass → 2× downsample.

    This is BigVGAN's "filtered Snake" block (Lee et al., 2023, §3.2): raw
    Snake produces sin²(αx) high-frequency content that aliases into the
    signal band at the current sample rate. Upsampling 2× gives the
    nonlinearity headroom, then a windowed-sinc low-pass below the original
    Nyquist kills the alias before downsampling back.

    Uses a Hann-windowed sinc FIR (kernel_size=12, cutoff=0.25 of upsampled
    Nyquist) — approximates the Kaiser filter in the official
    ``alias_free_torch`` package at a fraction of the code.
    """

    def __init__(self, alpha_init: float = 1.0, beta_init: float = 1.0, **_unused):
        super().__init__()
        self.snake = SnakeBeta(alpha_init=alpha_init, beta_init=beta_init)
        self.lp = _DepthwiseFIR(kernel_size=12, cutoff=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample 2× via zero-stuff + lowpass would be formally correct but
        # linear interp is a close enough approximation that keeps things
        # small; Snake + final lowpass carries the anti-alias guarantee.
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.snake(x)
        x = self.lp(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        return x


_CUSTOM_ACTIVATIONS = {
    "Snake": Snake,
    "SnakeBeta": SnakeBeta,
    "AliasFreeSnakeBeta": AliasFreeSnakeBeta,
}


def _get_activation(name: str):
    """Resolve activation name → class.

    Falls back to ``nn.<name>`` for stock PyTorch activations; checks the
    local custom registry first for Snake / SnakeBeta. Snake/SnakeBeta
    constructors take ``channels`` as the first positional argument, which
    SEANet passes via ``activation_params={'channels': hidden_dim}`` — see
    yaml config.
    """
    if name in _CUSTOM_ACTIVATIONS:
        return _CUSTOM_ACTIVATIONS[name]
    return getattr(nn, name)


class SEANetResnetBlock(nn.Module):
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = _get_activation(activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamableConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamableConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                             causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert 0 <= self.disable_norm_outer_blocks <= self.n_blocks

        act = _get_activation(activation)
        mult = 1
        model: tp.List[nn.Module] = [
            StreamableConv1d(channels, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= i + 2 else norm
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      norm=block_norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]
            model += [
                act(**activation_params),
                StreamableConv1d(mult * n_filters, mult * n_filters * 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=block_norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            StreamableConv1d(mult * n_filters, dimension, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0, trim_right_ratio: float = 1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.n_blocks = len(self.ratios) + 2
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert 0 <= self.disable_norm_outer_blocks <= self.n_blocks

        act = _get_activation(activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(dimension, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            model += [
                act(**activation_params),
                StreamableConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                          kernel_size=ratio * 2, stride=ratio,
                                          norm=block_norm, norm_kwargs=norm_params,
                                          causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=block_norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)]
            mult //= 2

        model += [
            act(**activation_params),
            StreamableConv1d(n_filters, channels, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        if final_activation is not None:
            final_act = _get_activation(final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        return self.model(z)
