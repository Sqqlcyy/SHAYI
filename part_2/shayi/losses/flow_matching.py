"""
Rectified flow matching loss for SHAYI DiT.

We jointly generate (z_p, z_r, z_t). For each factor:
  z0 ~ N(0, I)
  zt = (1 - tau) * z0 + tau * z1
  target = z1 - z0
  v_hat  = model(zt)
  loss   = MSE(v_hat, target)

We weight factors by their variance so no single factor dominates.
"""

import torch
import torch.nn.functional as F


def sample_flow_pair(z1: torch.Tensor, tau: torch.Tensor):
    """
    z1: [...]  (any shape)
    tau: [B]   (broadcasted)
    """
    z0 = torch.randn_like(z1)
    # reshape tau to broadcast
    view = [tau.shape[0]] + [1] * (z1.ndim - 1)
    tau_b = tau.view(*view).to(z1.dtype)
    zt = (1 - tau_b) * z0 + tau_b * z1
    target = z1 - z0
    return z0, zt, target


def flow_matching_step(model, batch, weights=(1.0, 1.0, 1.0), cfg_drop_prob=0.1):
    """
    batch keys: z_p, z_r, z_t, cond_tokens, cond_mask, global_cond
    """
    z_p1 = batch["z_p"]
    z_r1 = batch["z_r"]
    z_t1 = batch["z_t"]

    B = z_p1.shape[0]
    device = z_p1.device
    tau = torch.rand(B, device=device)

    _, zp_tau, tgt_p = sample_flow_pair(z_p1, tau)
    _, zr_tau, tgt_r = sample_flow_pair(z_r1, tau)
    _, zt_tau, tgt_t = sample_flow_pair(z_t1, tau)

    drop_cond = (torch.rand(B, device=device) < cfg_drop_prob)

    v_p, v_r, v_t = model(
        zp_tau, zr_tau, zt_tau, tau,
        cond_tokens=batch.get("cond_tokens"),
        cond_mask=batch.get("cond_mask"),
        global_cond=batch.get("global_cond"),
        drop_cond=drop_cond,
    )

    w_p, w_r, w_t = weights
    loss_p = F.mse_loss(v_p, tgt_p)
    loss_r = F.mse_loss(v_r, tgt_r)
    loss_t = F.mse_loss(v_t, tgt_t)

    loss = w_p * loss_p + w_r * loss_r + w_t * loss_t
    logs = {
        "loss": loss.item(),
        "loss_p": loss_p.item(),
        "loss_r": loss_r.item(),
        "loss_t": loss_t.item(),
    }
    return loss, logs
