from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor


LossFunc = Callable[[Dict[str, Any], Any], Optional[Tensor]]


class LossComposer:

    def __init__(self):
        self._terms: List[Tuple[str, LossFunc, str, float]] = []

    def register(
        self,
        name: str,
        fn: LossFunc,
        weight_key: str,
        default: float = 0.0,
    ) -> "LossComposer":
        self._terms.append((name, fn, weight_key, default))
        return self

    def __call__(
        self,
        enc: Dict[str, Any],
        cfg: Any,
        **extra,
    ) -> Dict[str, Tensor]:
        device = next(
            (v.device for v in enc.values() if isinstance(v, Tensor)),
            torch.device("cpu"),
        )
        zero = torch.zeros((), device=device)
        total = zero.clone()
        out: Dict[str, Tensor] = {}

        for name, fn, wk, default in self._terms:
            w = float(getattr(cfg, wk, default))
            if w == 0.0:
                out[f"loss/{name}"] = zero
                continue
            try:
                val = fn(enc, cfg, **extra)
            except Exception:
                val = None
            if val is None:
                out[f"loss/{name}"] = zero
                continue
            out[f"loss/{name}"] = val.detach()
            total = total + w * val

        out["loss/total"] = total
        return out
