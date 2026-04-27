from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn


def _pool(x: Tensor) -> Tensor:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.mean(dim=-1)
    if x.ndim == 4:
        B, K, D, T = x.shape
        return x.reshape(B, K * D, T).mean(dim=-1)
    raise ValueError(
        f"probe input must be [B,D], [B,D,T], or [B,K,D,T], got {tuple(x.shape)}"
    )


@torch.no_grad()
def linear_probe_r2(
    z: Tensor,
    y: Tensor,
    *,
    ridge: float = 1e-3,
) -> Tensor:
    z = _pool(z).float()
    y = _pool(y).float()
    B = z.shape[0]
    if B < 2:
        return z.new_tensor(float("nan"))

    zc = z - z.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)

    D_z = zc.shape[1]
    I = torch.eye(D_z, device=z.device, dtype=z.dtype)
    gram = zc.t() @ zc + ridge * B * I         # [D_z, D_z]
    rhs = zc.t() @ yc                           # [D_z, D_y]
    try:
        W = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        W = torch.linalg.lstsq(gram, rhs).solution

    y_pred = zc @ W
    ss_res = ((yc - y_pred) ** 2).sum(dim=0)    # [D_y]
    ss_tot = (yc ** 2).sum(dim=0) + 1e-12        # [D_y]
    r2_per_dim = 1.0 - ss_res / ss_tot
    return r2_per_dim.mean().clamp(-1.0, 1.0)


@torch.no_grad()
def all_pair_leakage(
    latents: dict,
    aux: dict,
) -> dict:
    out = {}
    for zk, zv in latents.items():
        if zv is None:
            continue
        for ak, av in aux.items():
            if av is None:
                continue
            r2 = linear_probe_r2(zv, av)
            out[f"leakage/{zk}_to_{ak}_R2"] = float(r2.item()) if r2 == r2 else float("nan")
    return out


# ---------------------------------------------------------------------------
# MINE-based MI diagnostic (Belghazi 2018).
#
# Complements the linear R² probe: R² only catches *linear* dependence, so a
# latent that encodes an attribute via any nonlinearity slips past it. A MINE
# critic trained for N inner steps estimates I(z_a; z_b) and reveals that
# leakage. Pure diagnostic — inputs are detached, no gradient flows back to
# the encoder.
# ---------------------------------------------------------------------------
class _MineCritic(nn.Module):
    def __init__(self, dim_a: int, dim_b: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_a + dim_b, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.net(torch.cat([a, b], dim=-1)).squeeze(-1)


def mine_mi_nats(
    z_a: Tensor,
    z_b: Tensor,
    *,
    inner_steps: int = 200,
    hidden: int = 256,
    lr: float = 1e-4,
    seed: Optional[int] = None,
) -> float:
    """MINE MI estimate in nats between (pooled) z_a and z_b.

    A fresh critic is trained on the given batch for ``inner_steps``. Inputs
    are detached so no gradient reaches the encoder. Batch too small (<8)
    returns NaN — the marginal estimate is useless there.
    """
    z_a = _pool(z_a).detach().float()
    z_b = _pool(z_b).detach().float()
    B = z_a.shape[0]
    if B < 8:
        return float("nan")

    device = z_a.device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    critic = _MineCritic(z_a.shape[-1], z_b.shape[-1], hidden=hidden).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)

    with torch.enable_grad():
        for _ in range(inner_steps):
            idx = torch.randperm(B, generator=gen, device=device)
            joint = critic(z_a, z_b)
            marginal = critic(z_a, z_b[idx])
            # Donsker–Varadhan lower bound on MI.
            loss = -(joint.mean() - torch.logsumexp(marginal, dim=0) + torch.log(torch.tensor(float(B), device=device)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    with torch.no_grad():
        idx = torch.randperm(B, generator=gen, device=device)
        joint = critic(z_a, z_b)
        marginal = critic(z_a, z_b[idx])
        mi = joint.mean() - torch.logsumexp(marginal, dim=0) + torch.log(torch.tensor(float(B), device=device))
    return float(mi.item())


def all_pair_mi(
    latents: Dict[str, Optional[Tensor]],
    *,
    inner_steps: int = 200,
    hidden: int = 256,
) -> Dict[str, float]:
    """Pairwise MI diagnostic across non-null latents."""
    items = [(k, v) for k, v in latents.items() if v is not None]
    out: Dict[str, float] = {}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ka, va = items[i]
            kb, vb = items[j]
            mi = mine_mi_nats(va, vb, inner_steps=inner_steps, hidden=hidden)
            out[f"mi/{ka}_{kb}_nats"] = mi
    return out
