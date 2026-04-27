"""
SHAYI DiT / Latent Flow Transformer
-----------------------------------
Inputs:
    z_p: [B, 4, 128, T]   (pre-softmax logits from AE pitch head)
    z_r: [B, 64, T]       (rhythm latent, time-varying)
    z_t: [B, 128]         (global timbre latent)

    cond_tokens: [B, L1, C_cond]   local text tokens (chunk_fact)
    cond_mask:   [B, L1]           1=valid, 0=pad
    global_cond: [B, C_global]     optional, track-level vector

Output:
    velocity predictions for (z_p, z_r, z_t)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
#  Time / positional embeddings
# -----------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B], values in [0, 1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=device).float()
            / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # [B, dim]


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, n_pos: int, dim: int):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(1, n_pos, dim))
        nn.init.trunc_normal_(self.emb, std=0.02)

    def forward(self, n: int) -> torch.Tensor:
        return self.emb[:, :n]


# -----------------------------
#  AdaLN modulation
# -----------------------------

class AdaLN(nn.Module):
    """
    AdaLN-Zero style modulation.
    scale, shift, gate are predicted from (time_emb + optional global_cond).
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim),
        )
        # zero-init so the block starts as identity
        nn.init.zeros_(self.to_mod[-1].weight)
        nn.init.zeros_(self.to_mod[-1].bias)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        # x: [B, N, D], cond_vec: [B, cond_dim]
        h = self.norm(x)
        mod = self.to_mod(cond_vec)[:, None]  # [B, 1, 3D]
        scale, shift, gate = mod.chunk(3, dim=-1)
        h = h * (1 + scale) + shift
        return h, gate


# -----------------------------
#  Standard cross-attention
# -----------------------------

class CrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.to_q = nn.Linear(q_dim, q_dim)
        self.to_k = nn.Linear(kv_dim, q_dim)
        self.to_v = nn.Linear(kv_dim, q_dim)
        self.to_out = nn.Linear(q_dim, q_dim)
        self.dropout = dropout

    def forward(self, q, k, v, key_padding_mask=None):
        # q: [B, Nq, D], k/v: [B, Nk, D]
        B, Nq, D = q.shape
        B, Nk, _ = k.shape
        H = self.num_heads
        Hd = D // H

        q = self.to_q(q).reshape(B, Nq, H, Hd).transpose(1, 2)
        k = self.to_k(k).reshape(B, Nk, H, Hd).transpose(1, 2)
        v = self.to_v(v).reshape(B, Nk, H, Hd).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)
        if key_padding_mask is not None:
            # True = keep, False = mask out
            mask = key_padding_mask[:, None, None, :]  # [B,1,1,Nk]
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = attn @ v  # [B, H, Nq, Hd]
        out = out.transpose(1, 2).reshape(B, Nq, D)
        return self.to_out(out)


# -----------------------------
#  HRCA (Contribution)
# -----------------------------

class HRCA(nn.Module):
    """
    Hierarchical Routed Cross-Attention.
      - Y_global from track-level condition (single vector broadcast or tokens)
      - Y_local  from chunk-level tokens
      - gating  alpha = sigmoid(MLP(x))
      - out = alpha * Y_global + (1 - alpha) * Y_local
    """

    def __init__(self, q_dim, local_dim, global_dim, num_heads, dropout=0.0):
        super().__init__()
        self.local_attn = CrossAttention(q_dim, local_dim, num_heads, dropout)
        self.global_attn = CrossAttention(q_dim, q_dim, num_heads, dropout)

        self.global_proj = nn.Linear(global_dim, q_dim) if global_dim is not None else None

        self.gate = nn.Sequential(
            nn.Linear(q_dim, q_dim),
            nn.SiLU(),
            nn.Linear(q_dim, 1),
        )

    def forward(self, x, local_tokens, local_mask, global_vec):
        # x:            [B, N, D]
        # local_tokens: [B, L1, C_local]
        # global_vec:   [B, C_global] or None
        B, N, D = x.shape

        if local_tokens is not None:
            y_local = self.local_attn(x, local_tokens, local_tokens, key_padding_mask=local_mask)
        else:
            y_local = torch.zeros_like(x)

        if global_vec is not None and self.global_proj is not None:
            gtok = self.global_proj(global_vec)            # [B, D]
            gtok = gtok[:, None, :].expand(-1, 1, -1)      # [B, 1, D]
            gmask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            y_global = self.global_attn(x, gtok, gtok, key_padding_mask=gmask)
        else:
            y_global = torch.zeros_like(x)

        alpha = torch.sigmoid(self.gate(x))  # [B, N, 1]
        return alpha * y_global + (1 - alpha) * y_local


# -----------------------------
#  Transformer block
# -----------------------------

class SHAYIBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: int,           # for AdaLN (time + global_cond merged)
        local_cond_dim: int,
        global_cond_dim: Optional[int],
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        use_hrca: bool = False,
    ):
        super().__init__()
        self.use_hrca = use_hrca

        self.norm_self = AdaLN(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )

        self.norm_cross = AdaLN(hidden_dim, cond_dim)
        if use_hrca:
            self.cross = HRCA(
                q_dim=hidden_dim,
                local_dim=local_cond_dim,
                global_dim=global_cond_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.cross = CrossAttention(
                q_dim=hidden_dim,
                kv_dim=local_cond_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.norm_mlp = AdaLN(hidden_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_vec: torch.Tensor,
        local_tokens: Optional[torch.Tensor],
        local_mask: Optional[torch.Tensor],
        global_vec: Optional[torch.Tensor],
    ):
        # self-attention
        h, gate = self.norm_self(x, cond_vec)
        h, _ = self.self_attn(h, h, h)
        x = x + gate * h

        # cross-attention (vanilla or HRCA)
        h, gate = self.norm_cross(x, cond_vec)
        if self.use_hrca:
            h = self.cross(h, local_tokens, local_mask, global_vec)
        else:
            if local_tokens is not None:
                h = self.cross(h, local_tokens, local_tokens, key_padding_mask=local_mask)
            else:
                h = torch.zeros_like(h)
        x = x + gate * h

        # mlp
        h, gate = self.norm_mlp(x, cond_vec)
        h = self.mlp(h)
        x = x + gate * h

        return x


# -----------------------------
#  SHAYI DiT
# -----------------------------

@dataclass
class SHAYIDiTConfig:
    T: int = 861
    zp_K: int = 4
    zp_bins: int = 128
    zr_dim: int = 64
    zt_dim: int = 128

    hidden_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.0

    local_cond_dim: int = 768
    global_cond_dim: Optional[int] = 512
    use_hrca: bool = False


class SHAYIDiT(nn.Module):
    def __init__(self, cfg: SHAYIDiTConfig):
        super().__init__()
        self.cfg = cfg

        D = cfg.hidden_dim
        T = cfg.T

        # ---- Input projections ----
        self.in_p = nn.Linear(cfg.zp_K * cfg.zp_bins, D)
        self.in_r = nn.Linear(cfg.zr_dim, D)
        self.in_t = nn.Linear(cfg.zt_dim, D)

        # ---- Type embeddings ----
        self.type_p = nn.Parameter(torch.zeros(1, 1, D))
        self.type_r = nn.Parameter(torch.zeros(1, 1, D))
        self.type_t = nn.Parameter(torch.zeros(1, 1, D))
        for p in [self.type_p, self.type_r, self.type_t]:
            nn.init.trunc_normal_(p, std=0.02)

        # ---- Position embeddings (shared over p and r) ----
        self.pos_time = LearnedPositionalEmbedding(T, D)

        # ---- Time embedding ----
        self.time_emb = SinusoidalTimeEmbedding(D)
        self.time_mlp = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        # cond for AdaLN = time_emb (+ optional global)
        cond_dim = D
        if cfg.global_cond_dim is not None:
            self.global_adaln_proj = nn.Linear(cfg.global_cond_dim, D)
        else:
            self.global_adaln_proj = None

        # ---- Blocks ----
        self.blocks = nn.ModuleList([
            SHAYIBlock(
                hidden_dim=D,
                num_heads=cfg.num_heads,
                cond_dim=cond_dim,
                local_cond_dim=cfg.local_cond_dim,
                global_cond_dim=cfg.global_cond_dim,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                use_hrca=cfg.use_hrca,
            )
            for _ in range(cfg.depth)
        ])

        self.norm_out = nn.LayerNorm(D)

        # ---- Output projections ----
        self.out_p = nn.Linear(D, cfg.zp_K * cfg.zp_bins)
        self.out_r = nn.Linear(D, cfg.zr_dim)
        self.out_t = nn.Linear(D, cfg.zt_dim)

    # -------------------------
    # tokenize / untokenize
    # -------------------------

    def tokenize(self, z_p, z_r, z_t):
        """
        z_p: [B, 4, 128, T]
        z_r: [B, 64, T]
        z_t: [B, 128]
        returns H: [B, 1 + T + T, D]
        """
        B, K, Pbins, T = z_p.shape
        D = self.cfg.hidden_dim

        # z_p -> [B, T, K*Pbins]
        zp = z_p.permute(0, 3, 1, 2).reshape(B, T, K * Pbins)
        hp = self.in_p(zp) + self.type_p + self.pos_time(T)   # [B, T, D]

        # z_r -> [B, T, 64]
        zr = z_r.permute(0, 2, 1)
        hr = self.in_r(zr) + self.type_r + self.pos_time(T)   # [B, T, D]

        # z_t -> [B, 1, D]
        ht = self.in_t(z_t)[:, None, :] + self.type_t         # [B, 1, D]

        H = torch.cat([ht, hp, hr], dim=1)                    # [B, 1+T+T, D]
        return H

    def untokenize(self, H, T):
        """
        H: [B, 1 + T + T, D]
        returns velocity tuple (v_p, v_r, v_t) matching input shapes.
        """
        B = H.shape[0]
        D = self.cfg.hidden_dim

        ht = H[:, :1]
        hp = H[:, 1:1 + T]
        hr = H[:, 1 + T:1 + T + T]

        v_p = self.out_p(hp)                                # [B, T, K*Pbins]
        v_p = v_p.reshape(B, T, self.cfg.zp_K, self.cfg.zp_bins)
        v_p = v_p.permute(0, 2, 3, 1).contiguous()          # [B, K, Pbins, T]

        v_r = self.out_r(hr).permute(0, 2, 1).contiguous()  # [B, 64, T]
        v_t = self.out_t(ht).squeeze(1)                     # [B, 128]
        return v_p, v_r, v_t

    # -------------------------
    # forward
    # -------------------------

    def forward(
        self,
        z_p, z_r, z_t,                 # noisy latents
        tau,                           # [B] in [0,1]
        cond_tokens=None,              # [B, L1, C_local]
        cond_mask=None,                # [B, L1] (bool)
        global_cond=None,              # [B, C_global]
        drop_cond: Optional[torch.Tensor] = None,  # [B] bool, True => null prompt
    ):
        B = z_p.shape[0]
        T = z_p.shape[-1]

        # optional CFG dropout: zero-out cond for dropped samples
        if drop_cond is not None:
            mask = drop_cond[:, None, None]
            if cond_tokens is not None:
                cond_tokens = torch.where(mask, torch.zeros_like(cond_tokens), cond_tokens)
            if cond_mask is not None:
                # if dropped, mark all tokens as masked-out (so attn = 0 effectively)
                cond_mask = torch.where(drop_cond[:, None], torch.zeros_like(cond_mask), cond_mask)
            if global_cond is not None:
                gmask = drop_cond[:, None]
                global_cond = torch.where(gmask, torch.zeros_like(global_cond), global_cond)

        # AdaLN conditioning vector = time + optional global
        t_emb = self.time_mlp(self.time_emb(tau))
        if self.global_adaln_proj is not None and global_cond is not None:
            cond_vec = t_emb + self.global_adaln_proj(global_cond)
        else:
            cond_vec = t_emb

        # tokenize
        H = self.tokenize(z_p, z_r, z_t)

        # run blocks
        for blk in self.blocks:
            H = blk(
                H,
                cond_vec=cond_vec,
                local_tokens=cond_tokens,
                local_mask=cond_mask,
                global_vec=global_cond,
            )

        H = self.norm_out(H)
        v_p, v_r, v_t = self.untokenize(H, T)
        return v_p, v_r, v_t
