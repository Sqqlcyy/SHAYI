"""
HiFRA-DiT: Hierarchical Factor-Routed Attention with Tension Modulation.

Implements the four-phase attention mechanism described in the SHAYI paper:
  Phase 1: Asymmetric Factor Cross-Attention (AFCA)
           Structural backbone via DAG: t > r > p
           Each factor attends to itself and its parents.
  Phase 2: Bi-directional Flow-Time-Gated Tension Field
           Six pairwise tension signals Psi_{i->j} for i,j in {p,r,t}, i != j.
           Each tension is gated by a flow-time-dependent scalar g_{i->j}(tau).
  Phase 3: Tension-Modulated Per-Factor Condition Routing
           External conditions are routed into three factor-specific streams.
           Tension states generate attention biases that modulate how each
           factor attends to its dedicated condition stream.
  Phase 4: Residual Update with Contrastive Tension Stop-Gradient
           Main update absorbs AFCA + condition routing.
           Tension enters as a contrastive stop-gradient residual.

Token layout inside each block:
    H = [ h_t (1 token),  h_p (T tokens),  h_r (T tokens) ]
    total N = 1 + 2T
    We maintain a factor_index tensor to slice/scatter per-factor states.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#  Utilities: AdaLN, time embeddings, positional embeddings
#  (mostly reused from your existing SHAYI DiT)
# =====================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
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
        return emb


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, n_pos: int, dim: int):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(1, n_pos, dim))
        nn.init.trunc_normal_(self.emb, std=0.02)

    def forward(self, n: int) -> torch.Tensor:
        return self.emb[:, :n]


class AdaLN(nn.Module):
    """AdaLN-Zero modulation. Returns (modulated_x, gate)."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * hidden_dim),
        )
        nn.init.zeros_(self.to_mod[-1].weight)
        nn.init.zeros_(self.to_mod[-1].bias)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        h = self.norm(x)
        mod = self.to_mod(cond_vec)[:, None]
        scale, shift, gate = mod.chunk(3, dim=-1)
        h = h * (1 + scale) + shift
        return h, gate


# =====================================================================
#  Phase 1: AFCA  (Asymmetric Factor Cross-Attention)
# =====================================================================

def build_dag_mask(T: int, device, dtype=torch.bool) -> torch.Tensor:
    """
    Build a (N, N) boolean mask that encodes the factor-DAG
        t >- r >- p
    meaning: t has no parents; r has parent t; p has parents t, r.
    The token layout is: [t(1 token), p(T tokens), r(T tokens)].
    mask[i, j] = True means query i is ALLOWED to attend to key j.

    Rules:
        - Every factor can attend to itself (intra-factor self-attention).
        - p can attend to r and t.
        - r can attend to t.
        - t can only attend to t.
    """
    N = 1 + 2 * T
    mask = torch.zeros(N, N, dtype=dtype, device=device)

    t_slc = slice(0, 1)
    p_slc = slice(1, 1 + T)
    r_slc = slice(1 + T, 1 + 2 * T)

    # intra-factor
    mask[t_slc, t_slc] = True
    mask[p_slc, p_slc] = True
    mask[r_slc, r_slc] = True

    # r -> t (rhythm attends to timbre)
    mask[r_slc, t_slc] = True

    # p -> t, p -> r (pitch attends to timbre and rhythm)
    mask[p_slc, t_slc] = True
    mask[p_slc, r_slc] = True

    return mask


class AFCA(nn.Module):
    """
    Asymmetric Factor Cross-Attention.

    Implemented as a single multi-head attention over the concatenated
    [t, p, r] tokens, with a block-structured DAG mask that allows only
    child-to-parent and self attention.

    This replaces the 'self_attn' block in the baseline SHAYIBlock:
    it serves BOTH as factor self-attention AND as structured inter-factor
    communication.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, H: torch.Tensor, dag_mask: torch.Tensor) -> torch.Tensor:
        """
        H: [B, N, D]
        dag_mask: [N, N] bool, True = allowed
        """
        B, N, D = H.shape
        Hd = self.head_dim
        Hn = self.num_heads

        qkv = self.qkv(H).reshape(B, N, 3, Hn, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, Hn, N, Hd]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)
        # broadcast dag_mask over B, Hn
        attn = attn.masked_fill(~dag_mask[None, None], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = attn @ v                              # [B, Hn, N, Hd]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        return self.out(out)


# =====================================================================
#  Phase 2: Flow-Time-Gated Tension Field
# =====================================================================

class FactorPairAttention(nn.Module):
    """
    Computes Psi_{src -> dst} = CrossAttn(Q = dst, K = V = src).

    Inputs are per-factor states already extracted from H:
        src: [B, N_src, D]
        dst: [B, N_dst, D]

    Output shape: [B, N_dst, D]   (lives in dst's token space)

    Notes on timbre:
      When src = z_t (N_src = 1), this is broadcasting a global summary
      into per-time-step tension at dst.
      When dst = z_t (N_dst = 1), we are pooling src into a single
      tension token for z_t.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        B, Nq, D = dst.shape
        _, Nk, _ = src.shape
        Hn, Hd = self.num_heads, self.head_dim

        q = self.to_q(dst).reshape(B, Nq, Hn, Hd).transpose(1, 2)
        k = self.to_k(src).reshape(B, Nk, Hn, Hd).transpose(1, 2)
        v = self.to_v(src).reshape(B, Nk, Hn, Hd).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, Nq, D)
        return self.to_out(out)


class TensionField(nn.Module):
    """
    Phase 2. Computes six pairwise tension signals among {p, r, t},
    each gated by a flow-time-dependent scalar g_{src->dst}(tau).

    We use a learnable MLP over the sinusoidal-embedded tau to produce
    six gates. Gates are initialized so that:
        - DAG-forward tensions (t->r, t->p, r->p) are active at all tau,
          with a mild bias toward small tau (structure emerges early).
        - DAG-backward tensions (p->r, p->t, r->t) are active later,
          with a bias toward large tau (refinement).

    Each tension has shape matching the destination factor's token span.
    The aggregated tension T_k is the sum over j != k of g_{j->k} * Psi_{j->k}.
    """

    _PAIRS: List[Tuple[str, str]] = [
        ("t", "p"), ("t", "r"),     # forward
        ("r", "p"),                  # forward
        ("p", "r"), ("p", "t"),     # backward
        ("r", "t"),                  # backward
    ]

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # One pairwise attention per (src, dst) pair
        self.pair_attn = nn.ModuleDict({
            f"{s}2{d}": FactorPairAttention(hidden_dim, num_heads, dropout)
            for (s, d) in self._PAIRS
        })

        # Flow-time gating MLP: tau_emb -> [6] gate logits
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(self._PAIRS)),
        )

        # Initialize gate biases to encode DAG-forward early, backward late.
        # We can't directly depend on tau at init, so we just bias the
        # constant offsets: forward pairs start active, backward pairs start off.
        # The model can learn the tau dependence from data.
        with torch.no_grad():
            last = self.gate_mlp[-1]
            nn.init.zeros_(last.weight)
            forward = {("t", "p"), ("t", "r"), ("r", "p")}
            biases = []
            for (s, d) in self._PAIRS:
                biases.append(1.0 if (s, d) in forward else -1.0)
            last.bias.copy_(torch.tensor(biases))

    def _slice_factors(self, H: torch.Tensor, T: int):
        """
        Split H [B, 1+2T, D] into (ht, hp, hr).
            ht: [B, 1, D]
            hp: [B, T, D]
            hr: [B, T, D]
        """
        ht = H[:, :1]
        hp = H[:, 1:1 + T]
        hr = H[:, 1 + T:1 + 2 * T]
        return ht, hp, hr

    def forward(
        self,
        H: torch.Tensor,
        T: int,
        tau_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns aggregated tensions (Tau_p, Tau_r, Tau_t):
            Tau_p: [B, T, D]
            Tau_r: [B, T, D]
            Tau_t: [B, 1, D]
        """
        ht, hp, hr = self._slice_factors(H, T)
        factors = {"p": hp, "r": hr, "t": ht}

        # Compute six pairwise tensions
        psi = {}
        for (s, d) in self._PAIRS:
            key = f"{s}2{d}"
            psi[(s, d)] = self.pair_attn[key](factors[s], factors[d])
            # shape matches factors[d]

        # Flow-time gates [B, 6]
        gate_logits = self.gate_mlp(tau_emb)           # [B, 6]
        gates = torch.sigmoid(gate_logits)             # [B, 6]

        # Aggregate per destination factor
        Tau_by_dst = {"p": None, "r": None, "t": None}
        for i, (s, d) in enumerate(self._PAIRS):
            g = gates[:, i:i + 1, None]                # [B, 1, 1]
            contrib = g * psi[(s, d)]
            if Tau_by_dst[d] is None:
                Tau_by_dst[d] = contrib
            else:
                Tau_by_dst[d] = Tau_by_dst[d] + contrib

        # If any destination received zero contributions (shouldn't happen
        # with current PAIRS), fill with zeros of the correct shape.
        if Tau_by_dst["p"] is None:
            Tau_by_dst["p"] = torch.zeros_like(hp)
        if Tau_by_dst["r"] is None:
            Tau_by_dst["r"] = torch.zeros_like(hr)
        if Tau_by_dst["t"] is None:
            Tau_by_dst["t"] = torch.zeros_like(ht)

        return Tau_by_dst["p"], Tau_by_dst["r"], Tau_by_dst["t"]


# =====================================================================
#  Phase 3: Tension-Modulated Per-Factor Condition Routing
# =====================================================================

class PerFactorConditionRouter(nn.Module):
    """
    Projects a shared condition stream (local text tokens + optional global
    vector) into three factor-specific condition streams c^{(p)}, c^{(r)},
    c^{(t)}, each with the same length as the original local tokens.

    Optionally an orthogonality regularizer can be applied externally on
    the three projection matrices to encourage per-factor semantic
    specialization (this is the attach point for AYI-style equivariance
    supervision on the condition side).
    """

    def __init__(
        self,
        local_dim: int,
        hidden_dim: int,
        global_dim: Optional[int] = None,
    ):
        super().__init__()
        self.proj_local = nn.ModuleDict({
            k: nn.Linear(local_dim, hidden_dim) for k in ("p", "r", "t")
        })
        self.proj_global = None
        if global_dim is not None:
            self.proj_global = nn.ModuleDict({
                k: nn.Linear(global_dim, hidden_dim) for k in ("p", "r", "t")
            })

    def forward(
        self,
        local_tokens: Optional[torch.Tensor],
        global_cond: Optional[torch.Tensor],
    ):
        """
        Returns dict with keys in {p, r, t}:
            each value is (cond_kv [B, L_k, D],  cond_pad_mask [B, L_k]).
        If local_tokens is None and global_cond is None, returns None.
        """
        if local_tokens is None and global_cond is None:
            return None

        out = {}
        for k in ("p", "r", "t"):
            parts = []
            if local_tokens is not None:
                parts.append(self.proj_local[k](local_tokens))          # [B, L1, D]
            if global_cond is not None and self.proj_global is not None:
                g = self.proj_global[k](global_cond)[:, None, :]        # [B, 1, D]
                parts.append(g)
            cond_kv = torch.cat(parts, dim=1)                           # [B, L_k, D]
            out[k] = cond_kv
        return out


class TensionModulatedCrossAttention(nn.Module):
    """
    Cross-attention from a factor's hidden state to its dedicated
    condition stream, with an additive attention bias derived from the
    aggregated tension of that factor.

        B_k = MLP_bias(Tau_k)  @  W_bias(cond_kv)^T        # [B, Nq, Nk]
        Y_k = softmax(Q K^T / sqrt(d) + B_k) V

    This is the key mechanism by which inter-factor tension modulates
    external conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        # tension -> per-query bias vector (shared across heads)
        self.bias_q = nn.Linear(hidden_dim, hidden_dim)
        self.bias_k = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hq: torch.Tensor,            # [B, Nq, D]  (factor hidden)
        cond_kv: torch.Tensor,       # [B, Nk, D]  (per-factor condition)
        cond_pad_mask: Optional[torch.Tensor],  # [B, Nk], True=keep
        tension: torch.Tensor,       # [B, Nq, D]
    ) -> torch.Tensor:
        B, Nq, D = hq.shape
        _, Nk, _ = cond_kv.shape
        Hn, Hd = self.num_heads, self.head_dim

        q = self.to_q(hq).reshape(B, Nq, Hn, Hd).transpose(1, 2)
        k = self.to_k(cond_kv).reshape(B, Nk, Hn, Hd).transpose(1, 2)
        v = self.to_v(cond_kv).reshape(B, Nk, Hn, Hd).transpose(1, 2)

        # standard attention logits
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)   # [B, Hn, Nq, Nk]

        # tension-derived additive bias (broadcast over heads)
        bq = self.bias_q(tension)                          # [B, Nq, D]
        bk = self.bias_k(cond_kv)                          # [B, Nk, D]
        bias = torch.einsum("bqd,bkd->bqk", bq, bk) / math.sqrt(D)
        attn = attn + bias[:, None]                        # broadcast heads

        # padding mask
        if cond_pad_mask is not None:
            m = cond_pad_mask[:, None, None, :]            # [B,1,1,Nk]
            attn = attn.masked_fill(~m, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = attn @ v                                     # [B, Hn, Nq, Hd]
        out = out.transpose(1, 2).reshape(B, Nq, D)
        return self.to_out(out)


# =====================================================================
#  Phase 4: HiFRA Block (combines all four phases)
# =====================================================================

class HiFRABlock(nn.Module):
    """
    A single HiFRA transformer block.

        H -> AFCA  -> + (residual, AdaLN-gated)
          -> TensionField  -> Tau_{p,r,t}
          -> Per-factor TensionModulatedCrossAttention  -> Y_{p,r,t}
          -> + (residual, AdaLN-gated)
          -> Contrastive tension residual  (gamma * (sg(Tau) - Proj(Tau)))
          -> FFN  -> + (residual, AdaLN-gated)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        tension_residual_scale: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tension_residual_scale = tension_residual_scale

        # Phase 1: AFCA
        self.norm_afca = AdaLN(hidden_dim, cond_dim)
        self.afca = AFCA(hidden_dim, num_heads, dropout)

        # Phase 2: Tension Field
        self.norm_tension = AdaLN(hidden_dim, cond_dim)
        self.tension_field = TensionField(hidden_dim, num_heads, dropout)

        # Phase 3: Tension-modulated per-factor cross-attention
        self.norm_cross = AdaLN(hidden_dim, cond_dim)
        self.cross_attn = nn.ModuleDict({
            k: TensionModulatedCrossAttention(hidden_dim, num_heads, dropout)
            for k in ("p", "r", "t")
        })

        # Phase 4: contrastive tension projection (per factor)
        self.tension_proj = nn.ModuleDict({
            k: nn.Linear(hidden_dim, hidden_dim) for k in ("p", "r", "t")
        })
        for k in ("p", "r", "t"):
            nn.init.zeros_(self.tension_proj[k].weight)
            nn.init.zeros_(self.tension_proj[k].bias)

        # Phase 4: FFN
        self.norm_mlp = AdaLN(hidden_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    # ---- helpers ---------------------------------------------------

    @staticmethod
    def _slice(H: torch.Tensor, T: int):
        ht = H[:, :1]
        hp = H[:, 1:1 + T]
        hr = H[:, 1 + T:1 + 2 * T]
        return ht, hp, hr

    @staticmethod
    def _cat(ht, hp, hr):
        return torch.cat([ht, hp, hr], dim=1)

    # ---- forward ---------------------------------------------------

    def forward(
        self,
        H: torch.Tensor,                 # [B, 1+2T, D]
        T: int,
        cond_vec: torch.Tensor,          # [B, cond_dim]   (time + global for AdaLN)
        tau_emb: torch.Tensor,           # [B, D]          (for TensionField gates)
        dag_mask: torch.Tensor,          # [1+2T, 1+2T] bool
        cond_per_factor: Optional[dict], # {'p':kv, 'r':kv, 't':kv} or None
        cond_pad_mask: Optional[torch.Tensor],
        drop_cond: Optional[torch.Tensor] = None,
    ):
        # ---------- Phase 1: AFCA -------------------------------
        h, g = self.norm_afca(H, cond_vec)
        h = self.afca(h, dag_mask)
        H = H + g * h

        # ---------- Phase 2: Tension Field ----------------------
        h_for_tension, _ = self.norm_tension(H, cond_vec)
        Tau_p, Tau_r, Tau_t = self.tension_field(h_for_tension, T, tau_emb)

        # ---------- Phase 3: Per-factor tension-modulated cross-attn
        # Only run if there is any condition.
        if cond_per_factor is not None:
            h_for_cross, g_cross = self.norm_cross(H, cond_vec)
            ht, hp, hr = self._slice(h_for_cross, T)

            y_t = self.cross_attn["t"](ht, cond_per_factor["t"], cond_pad_mask, Tau_t)
            y_p = self.cross_attn["p"](hp, cond_per_factor["p"], cond_pad_mask, Tau_p)
            y_r = self.cross_attn["r"](hr, cond_per_factor["r"], cond_pad_mask, Tau_r)

            Y = self._cat(y_t, y_p, y_r)

            # CFG drop: zero out the cross-attn contribution for dropped samples
            if drop_cond is not None:
                keep = (~drop_cond).to(Y.dtype)[:, None, None]
                Y = Y * keep

            H = H + g_cross * Y

        # ---------- Phase 4: Contrastive Tension Residual -------
        # Delta_k = gamma * (sg(Tau_k) - Proj_k(Tau_k))
        ht, hp, hr = self._slice(H, T)
        gamma = self.tension_residual_scale

        def _delta(tau, proj):
            # stop-gradient on the target; the projection is trained to
            # explain the tension, and the remainder nudges the hidden state.
            return gamma * (tau.detach() - proj(tau))

        ht = ht + _delta(Tau_t, self.tension_proj["t"])
        hp = hp + _delta(Tau_p, self.tension_proj["p"])
        hr = hr + _delta(Tau_r, self.tension_proj["r"])
        H = self._cat(ht, hp, hr)

        # ---------- Phase 4: FFN --------------------------------
        h, g = self.norm_mlp(H, cond_vec)
        h = self.mlp(h)
        H = H + g * h

        return H


# =====================================================================
#  Full HiFRA-DiT
# =====================================================================

@dataclass
class HiFRADiTConfig:
    # Latent shapes
    T: int = 861
    zp_K: int = 4
    zp_bins: int = 128
    zr_dim: int = 64
    zt_dim: int = 128

    # Transformer
    hidden_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.0

    # Conditioning
    local_cond_dim: int = 768
    global_cond_dim: Optional[int] = 512

    # HiFRA
    tension_residual_scale: float = 0.1


class HiFRADiT(nn.Module):
    """
    SHAYI DiT with full HiFRA attention stack.

    Drop-in replacement for SHAYIDiT(use_hrca=True/False). Exposes the same
    forward signature so the trainer / flow-matching code can swap them.
    """

    def __init__(self, cfg: HiFRADiTConfig):
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

        # Global projection used for AdaLN cond_vec
        cond_dim = D
        if cfg.global_cond_dim is not None:
            self.global_adaln_proj = nn.Linear(cfg.global_cond_dim, D)
        else:
            self.global_adaln_proj = None

        # ---- Per-factor condition router (Phase 3) ----
        self.cond_router = PerFactorConditionRouter(
            local_dim=cfg.local_cond_dim,
            hidden_dim=D,
            global_dim=cfg.global_cond_dim,
        )

        # ---- Blocks ----
        self.blocks = nn.ModuleList([
            HiFRABlock(
                hidden_dim=D,
                num_heads=cfg.num_heads,
                cond_dim=cond_dim,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                tension_residual_scale=cfg.tension_residual_scale,
            )
            for _ in range(cfg.depth)
        ])

        self.norm_out = nn.LayerNorm(D)

        # ---- Output projections ----
        self.out_p = nn.Linear(D, cfg.zp_K * cfg.zp_bins)
        self.out_r = nn.Linear(D, cfg.zr_dim)
        self.out_t = nn.Linear(D, cfg.zt_dim)

        # Pre-built DAG mask (registered as buffer; will move with .to(device))
        self.register_buffer("_dag_mask", build_dag_mask(T, device="cpu"), persistent=False)

    # -------------------------
    # tokenize / untokenize
    # -------------------------

    def tokenize(self, z_p, z_r, z_t):
        """
        z_p: [B, 4, 128, T]
        z_r: [B, 64, T]
        z_t: [B, 128]
        returns H: [B, 1 + 2T, D]    (layout: [t, p, r])
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

        H = torch.cat([ht, hp, hr], dim=1)                    # [B, 1+2T, D]
        return H

    def untokenize(self, H, T):
        """
        H: [B, 1 + 2T, D]
        returns velocity tuple (v_p, v_r, v_t) matching input shapes.
        """
        B = H.shape[0]

        ht = H[:, :1]
        hp = H[:, 1:1 + T]
        hr = H[:, 1 + T:1 + 2 * T]

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
        drop_cond: Optional[torch.Tensor] = None,
    ):
        B = z_p.shape[0]
        T = z_p.shape[-1]
        device = z_p.device

        # ---- CFG dropout: blank-out condition for dropped samples ----
        if drop_cond is not None:
            mask_tok = drop_cond[:, None, None]
            if cond_tokens is not None:
                cond_tokens = torch.where(mask_tok, torch.zeros_like(cond_tokens), cond_tokens)
            if cond_mask is not None:
                cond_mask = torch.where(drop_cond[:, None], torch.zeros_like(cond_mask), cond_mask)
            if global_cond is not None:
                gmask = drop_cond[:, None]
                global_cond = torch.where(gmask, torch.zeros_like(global_cond), global_cond)

        # ---- Time / global conditioning for AdaLN ----
        t_emb_raw = self.time_emb(tau)             # [B, D]
        t_emb = self.time_mlp(t_emb_raw)           # [B, D], used for AdaLN
        if self.global_adaln_proj is not None and global_cond is not None:
            cond_vec = t_emb + self.global_adaln_proj(global_cond)
        else:
            cond_vec = t_emb

        # tau_emb that the TensionField MLP consumes (keep raw sinusoid here;
        # the block-internal MLP will re-transform it)
        tau_emb = t_emb_raw

        # ---- Per-factor condition router ----
        cond_per_factor = self.cond_router(cond_tokens, global_cond)

        # The pad mask is shared across factors because we stack
        # [local_tokens (len L1), (optional) global_token (len 1)].
        if cond_per_factor is not None:
            # recompute full-length mask
            parts_mask = []
            if cond_tokens is not None:
                if cond_mask is None:
                    parts_mask.append(torch.ones(B, cond_tokens.shape[1],
                                                 dtype=torch.bool, device=device))
                else:
                    parts_mask.append(cond_mask.bool())
            if global_cond is not None:
                parts_mask.append(torch.ones(B, 1, dtype=torch.bool, device=device))
            full_cond_mask = torch.cat(parts_mask, dim=1)
        else:
            full_cond_mask = None

        # ---- Tokenize latents ----
        H = self.tokenize(z_p, z_r, z_t)

        # ---- DAG mask on the right device ----
                # ---- DAG mask on the right device ----
        if T == self.cfg.T:
            dag_mask = self._dag_mask.to(device=device)
        else:
            # 如果序列被截短了，动态当场生成一个小号的 Mask！
            dag_mask = build_dag_mask(T, device=device)

        # ---- Blocks ----
        for blk in self.blocks:
            H = blk(
                H=H,
                T=T,
                cond_vec=cond_vec,
                tau_emb=tau_emb,
                dag_mask=dag_mask,
                cond_per_factor=cond_per_factor,
                cond_pad_mask=full_cond_mask,
                drop_cond=drop_cond,
            )

        H = self.norm_out(H)
        v_p, v_r, v_t = self.untokenize(H, T)
        return v_p, v_r, v_t


# =====================================================================
#  Smoke test
# =====================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = HiFRADiTConfig(
        T=128,           # use a short T for smoke test
        depth=2,
        hidden_dim=256,
        num_heads=8,
        local_cond_dim=256,
        global_cond_dim=64,
    )
    model = HiFRADiT(cfg).cuda() if torch.cuda.is_available() else HiFRADiT(cfg)
    device = next(model.parameters()).device

    B = 2
    T = cfg.T
    z_p = torch.randn(B, cfg.zp_K, cfg.zp_bins, T, device=device)
    z_r = torch.randn(B, cfg.zr_dim, T, device=device)
    z_t = torch.randn(B, cfg.zt_dim, device=device)
    tau = torch.rand(B, device=device)
    cond_tokens = torch.randn(B, 16, cfg.local_cond_dim, device=device)
    cond_mask = torch.ones(B, 16, dtype=torch.bool, device=device)
    global_cond = torch.randn(B, cfg.global_cond_dim, device=device)
    drop_cond = torch.zeros(B, dtype=torch.bool, device=device)
    drop_cond[0] = True

    v_p, v_r, v_t = model(
        z_p, z_r, z_t, tau,
        cond_tokens=cond_tokens,
        cond_mask=cond_mask,
        global_cond=global_cond,
        drop_cond=drop_cond,
    )

    print("v_p:", v_p.shape, "expected:", z_p.shape)
    print("v_r:", v_r.shape, "expected:", z_r.shape)
    print("v_t:", v_t.shape, "expected:", z_t.shape)
    assert v_p.shape == z_p.shape
    assert v_r.shape == z_r.shape
    assert v_t.shape == z_t.shape

    # backward
    loss = (v_p.pow(2).mean() + v_r.pow(2).mean() + v_t.pow(2).mean())
    loss.backward()
    print("backward OK, loss =", loss.item())
