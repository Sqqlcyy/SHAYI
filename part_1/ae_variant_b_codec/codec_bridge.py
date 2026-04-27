from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.seanet_wrap import build_decoder, build_encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _freeze(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


# Canonical DAC model aliases. Keeps the name-parsing logic centralized and
_DAC_MODEL_ALIASES = {
    "dac_44khz": "44khz", "dac_44k": "44khz", "dac_44.1k": "44khz", "44khz": "44khz",
    "dac_24khz": "24khz", "dac_24k": "24khz",                       "24khz": "24khz",
    "dac_16khz": "16khz", "dac_16k": "16khz",                       "16khz": "16khz",
}
_DAC_VALID_BITRATES = ("8kbps", "16kbps")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SeanetSmokeConfig:
    """SEANet smoke-fallback params — only consulted when codec_backend=='seanet'."""
    in_channels: int = 1
    codec_dim: int = 256                      # output channels of the smoke codec
    ratios: List[int] = field(default_factory=lambda: [8, 8, 8])
    n_filters: int = 32
    n_residual_layers: int = 3
    norm: str = "weight_norm"
    causal: bool = False


@dataclass
class CodecBridgeConfig:
    # ---- Backend selection ----
    codec_backend: str = "dac"                # "dac" | "seanet"
    codec_model: str = "dac_44khz"            # DAC model alias (see _DAC_MODEL_ALIASES)
    codec_bitrate: str = "8kbps"              # DAC bitrate: "8kbps" | "16kbps"
    codec_sample_rate: int = 44100
    freeze_codec: bool = True

    # None = auto-infer via 100ms probe on first forward. Using Optional
    # instead of sentinel 0 so "0" can't accidentally become a valid value.
    codec_dim: Optional[int] = None

    # SEANet smoke-only sub-config. Ignored when codec_backend='dac'.
    seanet: SeanetSmokeConfig = field(default_factory=SeanetSmokeConfig)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------
class _CodecBridge(nn.Module):

    def __init__(self, cfg: CodecBridgeConfig):
        super().__init__()
        self.cfg = cfg
        self.backend = cfg.codec_backend.lower()
        self.freeze_codec = bool(cfg.freeze_codec)
        self.codec_dim: int = 0  # finalized in backend-specific init

        if self.backend == "seanet":
            self._init_seanet(cfg.seanet)
        elif self.backend == "dac":
            self._init_dac(cfg)
        else:
            raise ValueError(
                f"Unsupported codec_backend {cfg.codec_backend!r}. "
                "Expected one of: 'dac', 'seanet'."
            )

    # ---- backend-specific init paths ----
    def _init_seanet(self, scfg: SeanetSmokeConfig) -> None:
        self.codec_dim = int(scfg.codec_dim)
        self.encoder = build_encoder(
            in_channels=scfg.in_channels, dimension=self.codec_dim,
            n_filters=scfg.n_filters, n_residual_layers=scfg.n_residual_layers,
            ratios=scfg.ratios, norm=scfg.norm, causal=scfg.causal,
        )
        self.decoder = build_decoder(
            out_channels=scfg.in_channels, dimension=self.codec_dim,
            n_filters=scfg.n_filters, n_residual_layers=scfg.n_residual_layers,
            ratios=scfg.ratios, norm=scfg.norm, causal=scfg.causal,
        )
        if self.freeze_codec:
            _freeze(self.encoder)
            _freeze(self.decoder)

    def _init_dac(self, cfg: CodecBridgeConfig) -> None:
        try:
            import dac as dac_pkg
        except ImportError as e:
            raise RuntimeError(
                "codec_backend='dac' requires the `descript-audio-codec` "
                "package. Install it on the server, or set codec_backend='seanet' "
                "for a smoke fallback."
            ) from e

        model_type = _DAC_MODEL_ALIASES.get(cfg.codec_model.lower())
        if model_type is None:
            raise ValueError(
                f"Unknown DAC model {cfg.codec_model!r}. Valid aliases: "
                f"{sorted(set(_DAC_MODEL_ALIASES))}"
            )
        if cfg.codec_bitrate not in _DAC_VALID_BITRATES:
            raise ValueError(
                f"Unknown DAC bitrate {cfg.codec_bitrate!r}. Valid: {_DAC_VALID_BITRATES}"
            )

        self.codec = dac_pkg.utils.load_model(
            model_type=model_type,
            model_bitrate=cfg.codec_bitrate,
        )
        if self.freeze_codec:
            _freeze(self.codec)
        self.codec_dim = cfg.codec_dim if cfg.codec_dim is not None else self._infer_codec_dim()

    # ---- codec_dim inference ----
    def _infer_codec_dim(self) -> int:
        
        candidates = [
            getattr(self.codec, "codec_dim", None),
            getattr(getattr(self.codec, "model", None), "codebook_dim", None),
            getattr(getattr(self.codec, "model", None), "latent_dim", None),
            getattr(getattr(self.codec, "encoder", None), "dimension", None),
        ]
        for v in candidates:
            if isinstance(v, int) and v > 0:
                return int(v)

        device = next(self.codec.parameters(), torch.empty(0)).device
        sr = int(getattr(self.codec, "sample_rate", self.cfg.codec_sample_rate))
        # 100 ms probe. Long enough to clear DAC's downsampling stride (~4096
        # samples at 44.1 kHz) with margin, short enough to infer quickly.
        n = max(2048, sr // 10)
        x = torch.zeros(1, int(getattr(self.codec, "channels", 1)), n, device=device)
        with torch.no_grad():
            c = self._encode_raw(x)
        return int(c.shape[1])

    # ---- low-level encode/decode (no freeze wrapping) ----
    def _encode_raw(self, x: Tensor) -> Tensor:
        if self.backend == "seanet":
            return self.encoder(x)
        # DAC: `encode` returns (z_q, codes, latents, commit_loss, codebook_loss).
        # We use the continuous post-RVQ latent z_q as the adapter target.
        z_q, _codes, _latents, _commit, _cb = self.codec.encode(x)
        return z_q

    # ---- public interface ----
    def encode_latent(self, x: Tensor) -> Tensor:
        if self.freeze_codec:
            with torch.no_grad():
                return self._encode_raw(x).detach()
        return self._encode_raw(x)

    def decode_latent(self, c_hat: Tensor) -> Tensor:
        if self.backend == "seanet":
            return self.decoder(c_hat)
        # DAC continuous-latent decode (symmetric to _encode_raw's z_q path).
        return self.codec.decode(c_hat)
