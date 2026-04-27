from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from losses import (
    Stage1ReconLoss,
    OrthoLoss,
    gaussian_kl_loss,
    variance_of_energy_loss,
    linear_probe_r2,
)
from losses.leakage import all_pair_mi
from transforms import build_aux_bundle


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainerConfig:
    # Optimization
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    accum_steps: int = 1              # gradient accumulation (effective batch = batch_size * accum_steps)
    warmup_steps: int = 1000
    max_steps: int = 100_000

    # Loss weights
    w_recon: float = 1.0
    w_ortho: float = 0.1
    ortho_form: str = "cross_cov"            # "cross_cov" | "hsic"
    w_kl: float = 1e-4                        # Stable-Audio tiny KL (shared base)
    kl_beta_p: float = 1.0
    kl_beta_r: float = 1.0
    kl_beta_t: float = 1.0
    w_moe_t: float = 1e-3                     # default on for z_t
    w_moe_r: float = 0.0                      # optional ablation
    moe_k: int = 8
    # B-Codec: keep fusion latent on codec/RVQ manifold. Non-zero default
    w_codec_distill: float = 1.0
    require_codec_distill: bool = True        # assert codec_hat/codec_target are produced for B_CODEC

    # Pitch adversary on z_t (gradient reversal). Requires the model to be
    # built with ``use_pitch_adversary=True`` and ``aux_pitch`` in the batch.
    w_adv_pitch_on_t: float = 0.0

    # Pitch anchor warmup (F2'): forward-KL between a CQT-derived pitch
    # prior and the mean-over-K z_p distribution, active only for the
    # first ``pitch_anchor_decay_end`` steps. Its sole purpose is to break
    # the softmax symmetry trap that otherwise keeps z_p at uniform; at
    # convergence the weight is 0 so z_p is driven purely by L_recon +
    # L_ortho + L_AY. See notes/claude_review/2026-04-13_open_questions.md
    # §I.F2'.
    w_pitch_anchor: float = 0.0
    pitch_anchor_warmup: int = 1500        # full-strength steps
    pitch_anchor_decay_end: int = 5000     # linearly decay to 0 by this step
    pitch_anchor_tau: float = 0.5          # softmax temperature on CQT target; <1 = peakier

    # anchor loss for timbre
    w_timbre_anchor: float = 0.0
    timbre_anchor_warmup: int = 1500
    timbre_anchor_decay_end: int = 5000

    # rhythm anchor: symmetric anchor MSE(pool(z_r), pool(rhythm_multi)).
    # Keeps z_r's representation anchored to the rhythm stream through training.
    w_rhythm_anchor: float = 0.0
    rhythm_anchor_warmup: int = 1500
    rhythm_anchor_decay_end: int = 5000

    # ---- Variant-specific fields (documented but not yet split into subclasses) ----
    # B-only: w_codec_distill, require_codec_distill, w_trunk_l1, trunk_l1_form, aux_dropout_p
    # A-only: w_content_adv, w_adv_pitch_on_zt, w_adv_rhythm_on_zt, w_adv_pitch_on_zr, w_adv_envelope_on_zr
    # Both:   w_recon, w_ortho, w_kl, w_pitch_anchor, w_timbre_anchor, w_rhythm_anchor
    # See _validate_cfg_for_variant() in run.py for cross-variant checks.

    # GRL adversary weights (in-model adversaries populate enc dict)
    w_content_adv: float = 0.0
    w_adv_pitch_on_zt: float = 0.0
    w_adv_rhythm_on_zt: float = 0.0
    # v2 additions: adversaries on z_r
    w_adv_pitch_on_zr: float = 0.0
    w_adv_envelope_on_zr: float = 0.0

    # ---- v2-only loss terms ----
    # L1 sparsity on per-head trunk projector input weights (LOEV++-style
    # interpretable bottleneck). Set to 0 to disable.
    w_trunk_l1: float = 0.0
    # Sparsity form: "l1" (classical elementwise, legacy) or "group_lasso"
    # (per-input-channel — recommended for v2 because only group sparsity
    # gives the "trunk reads a subset of DAC channels" bottleneck claim).
    trunk_l1_form: str = "group_lasso"

    # Per-sample, per-head aux dropout 
    aux_dropout_p: float = 0.0

    # Latent-leakage probe cadence. 0 disables the closed-form R² probe.
    leakage_probe_every: int = 200

    # MINE-based MI diagnostic between latent pairs (z_p, z_r, z_t). Much
    # more expensive than the R² probe (trains a fresh critic each call) so
    # the cadence is larger by default. 0 disables.
    mi_probe_every: int = 2000
    mi_inner_steps: int = 200
    mi_hidden: int = 256

    # Reconstruction config
    sample_rate: int = 32000
    fft_sizes: Tuple[int, ...] = (1024, 2048, 4096)
    hop_sizes: Tuple[int, ...] = (120, 240, 480)
    win_lengths: Tuple[int, ...] = (960, 1920, 3840)
    n_mels: int = 128
    w_logmel: float = 0.5
    w_wav_l1: float = 1.0                     # time-domain L1 — pins phase

    # Logging / checkpointing
    log_every: int = 50
    val_every: int = 1000
    ckpt_every: int = 2000
    media_every: int = 1000
    out_dir: Path = Path("./runs/stage1")
    keep_last: int = 3

    # WandB
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"                # "online" | "offline" | "disabled"

    # TensorBoard
    tb_enabled: bool = True                   # writes to out_dir/tb/
    tb_flush_every: int = 20                  # steps between flushes

    # Device
    device: str = "cuda"
    amp: bool = True                          # bf16 autocast on cuda

    # Misc
    seed: int = 0


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def _warmup_cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Stage1Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.is_variant_b = bool(getattr(model, "uses_aux_adapters", False))
        self.input_mode = getattr(model, "input_mode", "waveform")
        self.needs_target_wav = bool(getattr(model, "needs_target_wav", False))

        # Losses
        self.recon_loss = Stage1ReconLoss(
            sample_rate=cfg.sample_rate,
            fft_sizes=list(cfg.fft_sizes),
            hop_sizes=list(cfg.hop_sizes),
            win_lengths=list(cfg.win_lengths),
            n_mels=cfg.n_mels,
            w_logmel=cfg.w_logmel,
            w_wav_l1=cfg.w_wav_l1,
        ).to(self.device)
        self.ortho_loss = OrthoLoss(form=cfg.ortho_form).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        # AMP
        self.amp_dtype = torch.bfloat16 if cfg.amp and self.device.type == "cuda" else torch.float32

        # Book-keeping
        self.global_step = 0
        self.best_val_recon = float("inf")
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # WandB
        self._wandb = None
        self._init_wandb()

        # TensorBoard
        self._tb = None
        self._init_tb()

    # ------------------------------------------------------------------
    def _init_wandb(self):
        if self.cfg.wandb_project is None or self.cfg.wandb_mode == "disabled":
            return
        try:
            import wandb
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=self.cfg.wandb_run_name,
                mode=self.cfg.wandb_mode,
                config=self._serialize_cfg(),
                dir=str(self.out_dir),
            )
            self._wandb = wandb
        except Exception as e:  # noqa: BLE001
            print(f"[trainer] WandB init failed ({e}); logging locally only.")
            self._wandb = None

    def _init_tb(self):
        if not self.cfg.tb_enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.out_dir / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(log_dir=str(tb_dir))
        except Exception as e:  # noqa: BLE001
            print(f"[trainer] TensorBoard init failed ({e}); continuing without TB.")
            self._tb = None

    def _serialize_cfg(self) -> Dict[str, Any]:
        out = {}
        for k, v in self.cfg.__dict__.items():
            if isinstance(v, Path):
                out[k] = str(v)
            elif isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------
    def _forward_model(
        self,
        batch: Dict[str, Tensor],
        *,
        deterministic: bool = False,
    ) -> Dict[str, Tensor]:
        audio = batch["audio"].to(self.device, non_blocking=True)
        if self.input_mode == "feature":
            feat_pitch = batch["feat_pitch"].to(self.device, non_blocking=True)
            feat_rhythm = batch["feat_rhythm"].to(self.device, non_blocking=True)
            feat_mel = batch["feat_mel"].to(self.device, non_blocking=True)
            # Option-B models consume envelope as 4th positional arg.
            wants_env = bool(getattr(self.model, "uses_envelope_aux", False))
            feat_env = batch["aux_timbre"].to(self.device, non_blocking=True) if wants_env else None
            extra_args = (feat_env,) if wants_env else ()
            # Models that use in-model GRL adversaries need aux tensors routed
            # as kwargs. Variant A-Codec additionally needs audio for the
            # codec_distill target.
            is_a_codec = type(self.model).__name__ == "VariantACodec"
            wants_adv_kwargs = is_a_codec or bool(getattr(self.model, "uses_adv_kwargs", False))
            extra_kwargs: Dict[str, Tensor] = {}
            if wants_adv_kwargs:
                extra_kwargs["aux_pitch"] = batch["aux_pitch"].to(self.device, non_blocking=True)
                extra_kwargs["aux_rhythm"] = batch["aux_rhythm"].to(self.device, non_blocking=True)
                extra_kwargs["aux_timbre"] = batch["aux_timbre"].to(self.device, non_blocking=True)
            if is_a_codec:
                extra_kwargs["audio"] = audio
            if self.needs_target_wav:
                enc = self.model(
                    feat_pitch, feat_rhythm, feat_mel, *extra_args, audio,
                    deterministic=deterministic, **extra_kwargs,
                )
            else:
                enc = self.model(
                    feat_pitch, feat_rhythm, feat_mel, *extra_args,
                    deterministic=deterministic, **extra_kwargs,
                )
            # Stash aux tensors so pitch_anchor / leakage / MI probes all
            # work in feature mode (they look up enc["__aux_*"]).
            if "aux_pitch" in batch:
                enc["__aux_pitch"] = batch["aux_pitch"].to(self.device, non_blocking=True)
            if "aux_rhythm" in batch:
                enc["__aux_rhythm"] = batch["aux_rhythm"].to(self.device, non_blocking=True)
            if "aux_timbre" in batch:
                enc["__aux_timbre"] = batch["aux_timbre"].to(self.device, non_blocking=True)
        elif self.is_variant_b:
            aux_bundle = build_aux_bundle(
                batch,
                device=self.device,
                is_variant_b=True,
                aux_zero_mask=getattr(self.model.cfg, "aux_zero_mask", ()),
            )
            aux_args = list(aux_bundle.encode_args())
            # Music ControlNet per-sample per-head aux dropout. Each aux
            # stream (pitch / rhythm / timbre) is independently zeroed with
            # probability ``aux_dropout_p`` for the whole batch sample.
            # Training-only: deterministic eval keeps aux intact.
            if self.model.training and self.cfg.aux_dropout_p > 0.0:
                aux_args = self._aux_dropout(aux_args, p=self.cfg.aux_dropout_p)
            enc = self.model(audio, *aux_args, deterministic=deterministic)
            # Stash aux_pitch for the pitch adversary / leakage probe — it's
            # only available here (the model itself does not see the raw CQT).
            if "aux_pitch" in batch:
                enc["__aux_pitch"] = batch["aux_pitch"].to(self.device, non_blocking=True)
            if "aux_rhythm" in batch:
                enc["__aux_rhythm"] = batch["aux_rhythm"].to(self.device, non_blocking=True)
            if "aux_timbre" in batch:
                enc["__aux_timbre"] = batch["aux_timbre"].to(self.device, non_blocking=True)
        else:
            enc = self.model(audio)
        enc["__audio"] = audio
        return enc

    # ------------------------------------------------------------------
    def _aux_dropout(self, aux_args: List[Tensor], p: float) -> List[Tensor]:
        """Music-ControlNet per-sample per-head aux dropout.

        For each of the three aux streams and each batch sample, draw an
        independent Bernoulli(p) mask. Masked samples have their aux
        tensor zeroed for that head. This forces the factor head to NOT
        rely exclusively on the aux shortcut — the DAC-trunk path has to
        carry real information on dropped samples.

        Args:
            aux_args: list of 3 aux tensors ``[B, D_aux, T]`` in order
                ``(aux_pitch, aux_rhythm, aux_timbre)``.
            p: per-sample per-head dropout probability (0.0 disables).

        Returns:
            New list of 3 tensors with the same shapes but with masked
            samples zeroed.
        """
        out: List[Tensor] = []
        for a in aux_args:
            if a is None:
                out.append(a)
                continue
            B = a.shape[0]
            # Bernoulli(p) per sample; keep_mask=1 when NOT dropped.
            keep = (torch.rand(B, device=a.device) > p).to(a.dtype)  # [B]
            shape = [B] + [1] * (a.ndim - 1)
            out.append(a * keep.view(*shape))
        return out

    def _compute_losses(self, enc: Dict[str, Tensor]) -> Dict[str, Tensor]:
        y = enc["__audio"]
        y_hat = enc["x_hat"]

        # align lengths (SEANet can off-by-ratio at the edge)
        T = min(y.shape[-1], y_hat.shape[-1])
        y = y[..., :T]
        y_hat = y_hat[..., :T]

        recon = self.recon_loss(y_hat, y)                          # dict

        # Skip ortho at B<2 (cross-cov/HSIC is noise at batch=1)
        B = y.shape[0]
        if B >= 2 and self.cfg.w_ortho > 0.0:
            ortho = self.ortho_loss(enc["z_p"], enc["z_r"], enc["z_t"])
        else:
            ortho = {"ortho/total": y.new_zeros(()), "ortho/pr": y.new_zeros(()),
                     "ortho/pt": y.new_zeros(()), "ortho/rt": y.new_zeros(())}

        # Per-factor KL (still Stable-Audio-tiny baseline; factor multipliers
        # let the anti-leak config make timbre tighter without disturbing
        # pitch/rhythm).
        # Skip KL when w_kl=0 (Variant A is deterministic, KL would push z→0)
        if self.cfg.w_kl > 0.0:
            kl_p = gaussian_kl_loss(enc["mu_p"], enc["log_var_p"])
            kl_r = gaussian_kl_loss(enc["mu_r"], enc["log_var_r"])
            kl_t = gaussian_kl_loss(enc["mu_t"], enc["log_var_t"])
        else:
            kl_p = kl_r = kl_t = y.new_zeros(())
        kl_total_weighted = (
            self.cfg.kl_beta_p * kl_p
            + self.cfg.kl_beta_r * kl_r
            + self.cfg.kl_beta_t * kl_t
        )

        moe_t = y.new_zeros(())
        moe_r = y.new_zeros(())
        if self.cfg.w_moe_t > 0.0:
            moe_t = variance_of_energy_loss(enc["z_t"], self.cfg.moe_k)
        if self.cfg.w_moe_r > 0.0:
            moe_r = variance_of_energy_loss(enc["z_r"], self.cfg.moe_k)
        moe_total = self.cfg.w_moe_t * moe_t + self.cfg.w_moe_r * moe_r

        codec_distill = y.new_zeros(())
        has_codec = ("codec_hat" in enc and "codec_target" in enc)
        if self.is_variant_b and self.cfg.require_codec_distill and not has_codec:
            raise RuntimeError(
                "B_CODEC forward did not produce codec_hat/codec_target. "
                "This is the load-bearing safety term that keeps c_hat on the "
                "frozen DAC manifold (see MODEL_MAINLINE.md). Set "
                "require_codec_distill=False only if you are deliberately "
                "running an ablation without it."
            )
        if self.cfg.w_codec_distill > 0.0 and has_codec:
            c_hat = enc["codec_hat"]
            c_tgt = enc["codec_target"].detach()
            T_c = min(c_hat.shape[-1], c_tgt.shape[-1])
            codec_distill = F.mse_loss(c_hat[..., :T_c], c_tgt[..., :T_c])

        # Pitch adversary on z_t (GRL). 
        adv_pitch_on_t = y.new_zeros(())
        adv_module = getattr(self.model, "pitch_adversary", None)
        if (
            self.cfg.w_adv_pitch_on_t > 0.0
            and adv_module is not None
            and "__aux_pitch" in enc
        ):
            adv_pitch_on_t = adv_module(enc["z_t"], enc["__aux_pitch"])

        # ---- v2-only loss terms ----
        # (1) Trunk L1 sparsity: encourages each per-head trunk projector
        #     to use only a sparse subset of codec channels (interpretable +
        #     extra bottleneck against leakage). Skipped if the model
        #     doesn't have trunk_p / trunk_r / trunk_t
        trunk_l1 = y.new_zeros(())
        if self.cfg.w_trunk_l1 > 0.0:
            trunks = [
                getattr(self.model, name, None)
                for name in ("trunk_p", "trunk_r", "trunk_t")
            ]
            trunks = [t for t in trunks if t is not None and hasattr(t, "input_channel_usage")]
            if trunks:
                if self.cfg.trunk_l1_form == "group_lasso":
                    trunk_l1 = sum(t.input_channel_group_lasso().sum() for t in trunks)
                elif self.cfg.trunk_l1_form == "l1":
                    trunk_l1 = sum(t.input_channel_usage().sum() for t in trunks)
                else:
                    raise ValueError(
                        f"Unknown trunk_l1_form: {self.cfg.trunk_l1_form!r} "
                        "(expected 'group_lasso' or 'l1')"
                    )

        # (2) Pitch anchor warmup (F2'): forward-KL from a CQT-derived prior
        #     to mean-over-K z_p. See TrainerConfig.w_pitch_anchor comments.
        pitch_anchor = y.new_zeros(())
        w_anchor = self._pitch_anchor_weight()
        if (
            w_anchor > 0.0
            and "__aux_pitch" in enc
            and enc.get("z_p") is not None
            and enc["z_p"].ndim == 4
        ):
            pitch_anchor = self._pitch_anchor_loss(enc["z_p"], enc["__aux_pitch"])

        # (3) Timbre anchor: linear-readout MSE z_t -> pool(envelope).
        timbre_anchor = y.new_zeros(())
        w_t_anchor = self._timbre_anchor_weight()
        if (
            w_t_anchor > 0.0
            and "__aux_timbre" in enc
            and enc.get("z_t") is not None
        ):
            timbre_anchor = self._timbre_anchor_loss(enc["z_t"], enc["__aux_timbre"])

        # (3b) Rhythm anchor: linear-readout MSE pool(z_r) -> pool(aux_rhythm).
        rhythm_anchor = y.new_zeros(())
        w_r_anchor = self._rhythm_anchor_weight()
        if (
            w_r_anchor > 0.0
            and "__aux_rhythm" in enc
            and enc.get("z_r") is not None
        ):
            rhythm_anchor = self._rhythm_anchor_loss(enc["z_r"], enc["__aux_rhythm"])

        # ---- Dict-based loss aggregation (audiocraft Balancer pattern) ----
        # Each loss term → {name: (value, weight)}. Adding/removing a loss
        # = adding/removing one entry, not editing a 15-line sum.
        def _enc_scalar(key):
            val = enc.get(key, y.new_zeros(()))
            return val if isinstance(val, Tensor) else y.new_zeros(())

        loss_terms = {
            "recon":             (recon["recon/total"],      self.cfg.w_recon),
            "ortho":             (ortho["ortho/total"],      self.cfg.w_ortho),
            "kl":                (kl_total_weighted,         self.cfg.w_kl),
            "moe":               (moe_total,                 1.0),  # moe_total already weighted internally
            "codec_distill":     (codec_distill,             self.cfg.w_codec_distill),
            "adv_pitch_on_t":    (adv_pitch_on_t,            self.cfg.w_adv_pitch_on_t),
            "trunk_l1":          (trunk_l1,                  self.cfg.w_trunk_l1),
            "pitch_anchor":      (pitch_anchor,              w_anchor),
            "timbre_anchor":     (timbre_anchor,             w_t_anchor),
            "rhythm_anchor":     (rhythm_anchor,             w_r_anchor),
            "content_adv":       (_enc_scalar("loss/content_adv"),       self.cfg.w_content_adv),
            "adv_pitch_on_zt":   (_enc_scalar("loss/adv_pitch_on_zt"),   self.cfg.w_adv_pitch_on_zt),
            "adv_rhythm_on_zt":  (_enc_scalar("loss/adv_rhythm_on_zt"),  self.cfg.w_adv_rhythm_on_zt),
            "adv_pitch_on_zr":   (_enc_scalar("loss/adv_pitch_on_zr"),   self.cfg.w_adv_pitch_on_zr),
            "adv_envelope_on_zr":(_enc_scalar("loss/adv_envelope_on_zr"),self.cfg.w_adv_envelope_on_zr),
        }

        total = y.new_zeros(())
        for val, w in loss_terms.values():
            if w != 0.0:
                total = total + w * val

        # Build output from loss_terms + sub-loss breakdown
        out: Dict[str, Tensor] = {"loss/total": total}
        for name, (val, _) in loss_terms.items():
            out[f"loss/{name}"] = val.detach() if isinstance(val, Tensor) else torch.tensor(float(val))
        # Recon sub-losses
        for k in ("recon/sc", "recon/log_mag", "recon/log_mel", "recon/wav_l1"):
            if k in recon:
                out[k] = recon[k]
        # Ortho sub-losses
        for k in ("ortho/pr", "ortho/pt", "ortho/rt"):
            if k in ortho:
                out[k] = ortho[k]
        # KL breakdown
        out["loss/kl_p"] = kl_p.detach() if isinstance(kl_p, Tensor) else torch.tensor(0.0)
        out["loss/kl_r"] = kl_r.detach() if isinstance(kl_r, Tensor) else torch.tensor(0.0)
        out["loss/kl_t"] = kl_t.detach() if isinstance(kl_t, Tensor) else torch.tensor(0.0)
        # Curriculum weights
        out["w/pitch_anchor"] = torch.tensor(float(w_anchor))
        out["w/timbre_anchor"] = torch.tensor(float(w_t_anchor))
        out["w/rhythm_anchor"] = torch.tensor(float(w_r_anchor))
        return out

    # ------------------------------------------------------------------
    def _timbre_anchor_weight(self) -> float:
        """Curriculum for z_t ← envelope anchor (symmetric to pitch anchor)."""
        if self.cfg.w_timbre_anchor <= 0.0:
            return 0.0
        s = self.global_step
        if s < self.cfg.timbre_anchor_warmup:
            return float(self.cfg.w_timbre_anchor)
        if s >= self.cfg.timbre_anchor_decay_end:
            return 0.0
        span = max(self.cfg.timbre_anchor_decay_end - self.cfg.timbre_anchor_warmup, 1)
        r = (s - self.cfg.timbre_anchor_warmup) / span
        return float(self.cfg.w_timbre_anchor) * (1.0 - r)

    def _timbre_anchor_loss(self, z_t: Tensor, aux_timbre: Tensor) -> Tensor:
        """Linear-readout MSE: force z_t to be predictive of envelope pool.

        A trainable linear head maps ``z_t -> pool_time(envelope)``; the MSE
        gradient flows back into z_t, pulling it into a representation from
        which the pitch-invariant envelope is linearly recoverable. The
        readout params are registered with the main optimizer on first use.
        """
        tgt = aux_timbre.mean(dim=-1).to(z_t.dtype).detach()   # [B, C_env]
        need_new = (
            not hasattr(self, "_timbre_anchor_head")
            or self._timbre_anchor_head is None
            or self._timbre_anchor_head.in_features != z_t.shape[-1]
            or self._timbre_anchor_head.out_features != tgt.shape[-1]
        )
        if need_new:
            self._timbre_anchor_head = nn.Linear(
                z_t.shape[-1], tgt.shape[-1], bias=True,
            ).to(z_t.device).to(z_t.dtype)
            self.optimizer.add_param_group({"params": list(self._timbre_anchor_head.parameters())})
        pred = self._timbre_anchor_head(z_t)
        return F.mse_loss(pred, tgt)

    # ------------------------------------------------------------------
    def _rhythm_anchor_weight(self) -> float:
        """Curriculum for z_r ← rhythm anchor (symmetric to pitch anchor)."""
        if self.cfg.w_rhythm_anchor <= 0.0:
            return 0.0
        s = self.global_step
        if s < self.cfg.rhythm_anchor_warmup:
            return float(self.cfg.w_rhythm_anchor)
        if s >= self.cfg.rhythm_anchor_decay_end:
            return 0.0
        span = max(self.cfg.rhythm_anchor_decay_end - self.cfg.rhythm_anchor_warmup, 1)
        r = (s - self.cfg.rhythm_anchor_warmup) / span
        return float(self.cfg.w_rhythm_anchor) * (1.0 - r)

    def _rhythm_anchor_loss(self, z_r: Tensor, aux_rhythm: Tensor) -> Tensor:
        """Linear-readout MSE: force pool(z_r) to be predictive of pool(rhythm).

        Same pattern as _timbre_anchor_loss: trainer-owned readout head,
        params added to the main optimizer on first use (lazy init).
        """
        # Pool over time
        z_pool = z_r.mean(dim=-1).to(z_r.dtype)     # [B, d_r]
        tgt = aux_rhythm.mean(dim=-1).to(z_r.dtype).detach()   # [B, C_rhythm]
        need_new = (
            not hasattr(self, "_rhythm_anchor_head")
            or self._rhythm_anchor_head is None
            or self._rhythm_anchor_head.in_features != z_pool.shape[-1]
            or self._rhythm_anchor_head.out_features != tgt.shape[-1]
        )
        if need_new:
            self._rhythm_anchor_head = nn.Linear(
                z_pool.shape[-1], tgt.shape[-1], bias=True,
            ).to(z_pool.device).to(z_pool.dtype)
            self.optimizer.add_param_group({"params": list(self._rhythm_anchor_head.parameters())})
        pred = self._rhythm_anchor_head(z_pool)
        return F.mse_loss(pred, tgt)

    # ------------------------------------------------------------------
    def _pitch_anchor_weight(self) -> float:
        """Curriculum for the F2' pitch anchor KL (warmup, then linear decay).

        * step <  pitch_anchor_warmup       → full strength w_pitch_anchor
        * step in [warmup, decay_end)       → linear decay to 0
        * step >= pitch_anchor_decay_end    → 0 (anchor is gone)
        """
        if self.cfg.w_pitch_anchor <= 0.0:
            return 0.0
        s = self.global_step
        if s < self.cfg.pitch_anchor_warmup:
            return float(self.cfg.w_pitch_anchor)
        if s >= self.cfg.pitch_anchor_decay_end:
            return 0.0
        span = max(self.cfg.pitch_anchor_decay_end - self.cfg.pitch_anchor_warmup, 1)
        r = (s - self.cfg.pitch_anchor_warmup) / span
        return float(self.cfg.w_pitch_anchor) * (1.0 - r)

    # ------------------------------------------------------------------
    def _pitch_anchor_loss(self, z_p: Tensor, aux_pitch: Tensor) -> Tensor:
        """Forward-KL ``KL(p_target || mean_K(z_p))`` with a CQT-derived target.

        ``z_p`` : ``[B, K, n_bins, T_z]`` softmax along the bin axis.
        ``aux_pitch`` : ``[B, n_cqt, T_aux]`` real-valued CQT salience.

        Returns a scalar. Forward KL is mode-covering: it pushes the
        aggregated pitch distribution ``mean_K(z_p)`` to have mass wherever
        the CQT prior has mass, without forcing it to ignore any peaks —
        exactly what we want for a *weak* symmetry-breaking anchor.

        Implementation notes:
        * The CQT bin axis (default 588) is resampled to ``n_bins`` (default
          128) via linear interpolation; this is a coarse projection but
          good enough for a warmup prior.
        * The target is a temperature-softmax of the resampled CQT along the
          bin axis with ``τ = pitch_anchor_tau`` (<1 makes it peakier so the
          anchor pulls harder toward genuine peaks).
        """
        # Resample CQT bin axis to n_bins. F.interpolate(mode='linear') only
        # resamples the last dim, so transpose (bin, time) first.
        n_bins = z_p.shape[2]
        aux_bins = aux_pitch.transpose(1, 2).contiguous()                        # [B, T_aux, n_cqt]
        aux_bins = F.interpolate(aux_bins, size=n_bins, mode="linear", align_corners=False)  # [B, T_aux, n_bins]
        aux_bins = aux_bins.transpose(1, 2).contiguous()                         # [B, n_bins, T_aux]
        # Align time with z_p (second interpolate goes along T, which is last).
        T_z = z_p.shape[-1]
        if aux_bins.shape[-1] != T_z:
            aux_bins = F.interpolate(aux_bins, size=T_z, mode="linear", align_corners=False)
        # Peakier softmax target.
        p_target = F.softmax(aux_bins / float(self.cfg.pitch_anchor_tau), dim=1)  # [B, n_bins, T]
        # Aggregate z_p over the K polyphony channels.
        z_agg = z_p.mean(dim=1)                                                   # [B, n_bins, T]
        eps = 1e-8
        log_z = torch.log(z_agg.clamp_min(eps))
        log_t = torch.log(p_target.clamp_min(eps))
        # KL(P || Q) = sum_bins P * (log P - log Q), averaged over (B, T).
        kl = (p_target * (log_t - log_z)).sum(dim=1).mean()
        return kl

    # ------------------------------------------------------------------
    def _leakage_stats(self, enc: Dict[str, Tensor]) -> Dict[str, float]:
        """Closed-form linear-probe R² from each z onto each aux stream.

        High ``z_t_to_pitch`` = the timbre head still linearly exposes pitch
        (the leak the anti-leak config is trying to close). High
        ``z_p_to_pitch`` = the pitch head is doing its job (positive control).
        """
        if self.cfg.leakage_probe_every <= 0:
            return {}
        if self.global_step % self.cfg.leakage_probe_every != 0:
            return {}
        out: Dict[str, float] = {}
        aux_pitch = enc.get("__aux_pitch")
        aux_rhythm = enc.get("__aux_rhythm")
        for zk in ("z_p", "z_r", "z_t"):
            z = enc.get(zk)
            if z is None:
                continue
            if aux_pitch is not None:
                r2 = linear_probe_r2(z, aux_pitch)
                out[f"leakage/{zk}_to_pitch_R2"] = float(r2.item()) if r2 == r2 else float("nan")
            if aux_rhythm is not None:
                r2 = linear_probe_r2(z, aux_rhythm)
                out[f"leakage/{zk}_to_rhythm_R2"] = float(r2.item()) if r2 == r2 else float("nan")
        return out

    # ------------------------------------------------------------------
    def _mi_stats(self, enc: Dict[str, Tensor]) -> Dict[str, float]:
        """MINE-based MI diagnostic between latent pairs.

        Catches nonlinear dependence the R² probe misses. Pure diagnostic:
        inputs detached, encoder parameters untouched.
        """
        if self.cfg.mi_probe_every <= 0:
            return {}
        if self.global_step % self.cfg.mi_probe_every != 0:
            return {}
        latents = {k: enc.get(k) for k in ("z_p", "z_r", "z_t")}
        return all_pair_mi(
            latents,
            inner_steps=self.cfg.mi_inner_steps,
            hidden=self.cfg.mi_hidden,
        )

    # ------------------------------------------------------------------
    def _latent_stats(self, enc: Dict[str, Tensor]) -> Dict[str, float]:
        """Dead-dim / aliveness metrics for each latent.

        For continuous latents (z_r ``[B,D,T]``, z_t ``[B,D]``) we use the
        classical "fraction of dims with std > 1e-2" proxy, plus mean-abs.

        For the v2 structured pitch latent ``z_p [B, K, n_bins, T]`` that
        naive metric is misleading: z_p is a softmax distribution along the
        ``n_bins`` axis, so a perfectly uniform (dead) z_p has ``mean_abs ≈
        1 / n_bins`` and all per-dim stds are tiny — it looks "dead" by the
        std threshold even when the architecture is healthy, and it looks
        "alive" if you flip the threshold. Instead report two distribution-
        native metrics that make symmetry-collapse unambiguous:

          * ``z_p/entropy_norm``: mean entropy of z_p over the bin axis,
            normalized by log(n_bins). Uniform → 1.0, one-hot → 0.0.
            A healthy pitch head with real peaks should be < 0.9.

          * ``z_p/peak_mean``: average of max-bin probability per (batch, K,
            frame). Uniform → 1/n_bins (~0.008 at n_bins=128), one-hot → 1.0.
            A healthy head typically ≥ 0.05.
        """
        out = {}
        for key in ("z_p", "z_r", "z_t"):
            z = enc[key].detach()
            if key == "z_p" and z.ndim == 4:
                # [B, K, n_bins, T] softmax distribution along bin axis.
                eps = 1e-8
                H = -(z.clamp_min(eps) * z.clamp_min(eps).log()).sum(dim=2)  # [B, K, T]
                log_nbins = float(torch.log(torch.tensor(z.shape[2], dtype=z.dtype)))
                out["z_p/entropy_norm"] = float((H / max(log_nbins, eps)).mean().item())
                out["z_p/peak_mean"] = float(z.max(dim=2).values.mean().item())
                # legacy fields kept for backward compat of log dashboards
                out["latent/mean_abs_z_p"] = float(z.abs().mean().item())
                # reinterpret dim_activity for softmax: fraction of (K, t) slots
                # whose entropy is noticeably below uniform (< 0.95 × max entropy)
                alive = (H < 0.95 * log_nbins).float().mean()
                out["latent/dim_activity_z_p"] = float(alive.item())
                continue
            if z.ndim == 3:
                std = z.std(dim=(0, 2), unbiased=False)
            else:
                std = z.std(dim=0, unbiased=False)
            out[f"latent/mean_abs_{key}"] = float(z.abs().mean().item())
            out[f"latent/dim_activity_{key}"] = float((std > 1e-2).float().mean().item())
        return out

    # ------------------------------------------------------------------
    def train_one_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.train()

        # LR schedule
        lr = _warmup_cosine_lr(
            self.global_step, self.cfg.lr, self.cfg.warmup_steps, self.cfg.max_steps
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation: only zero grad at start of accumulation window
        if self.global_step % self.cfg.accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=(self.amp_dtype != torch.float32),
        ):
            # Sync global_step to model for content warmup curriculum
            if hasattr(self.model, "_global_step"):
                self.model._global_step = self.global_step
            enc = self._forward_model(batch)
            losses = self._compute_losses(enc)

        total = losses["loss/total"]
        (total / self.cfg.accum_steps).backward()  # scale for gradient accumulation

        # Only step every accum_steps (effective batch = batch_size * accum_steps)
        if (self.global_step + 1) % self.cfg.accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.grad_clip
            )
            self.optimizer.step()
        else:
            grad_norm = torch.tensor(0.0)

        stats = {k: float(v.item()) if isinstance(v, Tensor) else float(v) for k, v in losses.items()}
        stats["lr"] = lr
        stats["grad_norm/total"] = float(grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm)
        stats.update(self._latent_stats(enc))
        stats.update(self._leakage_stats(enc))
        stats.update(self._mi_stats(enc))
        return stats

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        sums: Dict[str, float] = {}
        n = 0
        for batch in self.val_loader:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=(self.amp_dtype != torch.float32),
            ):
                enc = self._forward_model(batch)
                losses = self._compute_losses(enc)
            n += 1
            for k, v in losses.items():
                val = float(v.item()) if isinstance(v, Tensor) else float(v)
                sums[f"val/{k}"] = sums.get(f"val/{k}", 0.0) + val
        return {k: v / max(n, 1) for k, v in sums.items()}

    # ------------------------------------------------------------------
    def _log(self, stats: Dict[str, float]):
        stats = {"step": self.global_step, **stats}
        if self._wandb is not None:
            self._wandb.log(stats, step=self.global_step)
        if self._tb is not None:
            for k, v in stats.items():
                if k == "step":
                    continue
                try:
                    self._tb.add_scalar(k, float(v), self.global_step)
                except (TypeError, ValueError):
                    pass
            if self.global_step % max(self.cfg.tb_flush_every, 1) == 0:
                self._tb.flush()
        # Always append to local JSONL so rerunnable even without WandB.
        with open(self.out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")

    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag: str):
        path = self.out_dir / f"ckpt_{tag}.pt"
        state = {
            "step": self.global_step,
            "version": type(self.model).__name__,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_recon": self.best_val_recon,
            "cfg": self._serialize_cfg(),
        }
        # Save trainer-owned anchor readout heads (not in model state_dict).
        aux_heads = {}
        for name in ("_timbre_anchor_head", "_rhythm_anchor_head"):
            head = getattr(self, name, None)
            if head is not None:
                aux_heads[name] = head.state_dict()
        if aux_heads:
            state["trainer_aux_heads"] = aux_heads
        torch.save(state, path)
        self._cleanup_ckpts()
        return path

    def _cleanup_ckpts(self):
        ckpts = sorted(self.out_dir.glob("ckpt_step*.pt"))
        if len(ckpts) > self.cfg.keep_last:
            for p in ckpts[: -self.cfg.keep_last]:
                try:
                    p.unlink()
                except OSError:
                    pass

    def fit(self):
        print(f"[trainer] device={self.device} amp_dtype={self.amp_dtype} steps={self.cfg.max_steps}")
        print(
            f"[trainer] model={type(self.model).__name__} "
            f"input_mode={self.input_mode} variant_b={self.is_variant_b}"
        )
        print(f"[trainer] out_dir={self.out_dir}")

        t0 = time.time()
        it = iter(self.train_loader)
        while self.global_step < self.cfg.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)

            stats = self.train_one_step(batch)
            self.global_step += 1

            if self.global_step % self.cfg.log_every == 0:
                dt = time.time() - t0
                stats["time/sec_per_step"] = dt / max(self.cfg.log_every, 1)
                t0 = time.time()
                print(
                    f"step={self.global_step} "
                    f"loss={stats['loss/total']:.4f} "
                    f"recon={stats['loss/recon']:.4f} "
                    f"ortho={stats['loss/ortho']:.4e} "
                    f"kl={stats['loss/kl']:.4e} "
                    f"lr={stats['lr']:.2e}"
                )
                self._log(stats)

            if self.val_loader is not None and self.global_step % self.cfg.val_every == 0:
                val_stats = self.validate()
                self._log(val_stats)
                val_recon = val_stats.get("val/loss/recon", float("inf"))
                if val_recon < self.best_val_recon:
                    self.best_val_recon = val_recon
                    self._save_checkpoint("best")

            if self.global_step % self.cfg.ckpt_every == 0:
                self._save_checkpoint(f"step{self.global_step:08d}")

        # Final save
        self._save_checkpoint(f"step{self.global_step:08d}")
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
