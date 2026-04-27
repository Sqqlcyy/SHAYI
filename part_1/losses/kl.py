from __future__ import annotations

import torch
from torch import Tensor


def gaussian_kl_loss(mu: Tensor, log_var: Tensor) -> Tensor:
    # 0.5 * (mu^2 + sigma^2 - log sigma^2 - 1)
    kl = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1.0)
    return kl.mean()
