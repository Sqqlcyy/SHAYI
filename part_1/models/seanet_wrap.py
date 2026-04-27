from __future__ import annotations
from typing import List
import torch.nn as nn
from ._seanet_core import SEANetEncoder, SEANetDecoder


def build_encoder(
    in_channels: int = 1,
    dimension: int = 256,
    n_filters: int = 32,
    n_residual_layers: int = 3,
    ratios: List[int] = (8, 5, 4, 4),
    causal: bool = False,
    norm: str = "weight_norm",
    lstm: int = 0,
) -> nn.Module:
    return SEANetEncoder(
        channels=in_channels,
        dimension=dimension,
        n_filters=n_filters,
        n_residual_layers=n_residual_layers,
        ratios=list(ratios),
        causal=causal,
        norm=norm,
        lstm=lstm,
    )


def build_decoder(
    out_channels: int = 1,
    dimension: int = 256,
    n_filters: int = 32,
    n_residual_layers: int = 3,
    ratios: List[int] = (8, 5, 4, 4),
    causal: bool = False,
    norm: str = "weight_norm",
    lstm: int = 0,
    final_activation: str = None,
    activation: str = "ELU",
    activation_params: dict = None,
) -> nn.Module:
    kwargs = {}
    if activation_params is not None:
        kwargs["activation_params"] = activation_params
    return SEANetDecoder(
        channels=out_channels,
        dimension=dimension,
        n_filters=n_filters,
        n_residual_layers=n_residual_layers,
        ratios=list(ratios),
        causal=causal,
        norm=norm,
        lstm=lstm,
        final_activation=final_activation,
        activation=activation,
        **kwargs,
    )
