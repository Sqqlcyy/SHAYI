from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from losses import (
    CombinedDiscriminator,
    feature_matching_loss,
    hinge_d_loss,
    hinge_g_loss,
)
from stage1.trainer import Stage1Trainer, TrainerConfig, _warmup_cosine_lr


@dataclass
class AdvTrainerConfig(TrainerConfig):
    # Adversarial weights
    w_adv: float = 1.0
    w_feat_match: float = 10.0
    disc_lr: float = 3e-4
    disc_betas: tuple = (0.8, 0.9)
    disc_weight_decay: float = 0.0
    adv_start_step: int = 1000         # warmup reconstruction before D kicks in
    disc_periods: tuple = (2, 3, 5, 7, 11)
    disc_n_scales: int = 3
    disc_max_samples: int = 131_072
    disc_lr_decay: bool = True          # cosine decay disc lr (prevents D overpowering G)    # discriminator sees a shared random crop to keep MPD/MSD memory bounded

    # UnivNet / Fre-GAN style multi-scale STFT discriminator — catches
    # spectral artefacts the time-domain MPD+MSD miss.
    use_mssd: bool = False

    # EMA of generator weights — swapped in during validation / checkpoint.
    # HiFi-GAN / BigVGAN standard practice (~0.999 decay).
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_warmup: int = 1000             # start updating EMA only after this step


class Stage1AdvTrainer(Stage1Trainer):
    """Stage-1 trainer with HiFi-GAN style adversarial losses."""

    cfg: AdvTrainerConfig  # type: ignore[assignment]

    def __init__(
        self,
        cfg: AdvTrainerConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        super().__init__(cfg, model, train_loader, val_loader)

        self.discriminator = CombinedDiscriminator(
            periods=tuple(cfg.disc_periods),
            n_scales=int(cfg.disc_n_scales),
            use_mssd=bool(getattr(cfg, "use_mssd", False)),
        ).to(self.device)

        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=cfg.disc_lr,
            betas=tuple(cfg.disc_betas),
            weight_decay=cfg.disc_weight_decay,
        )

        # ----- EMA of generator weights -----
        # state_dict-based EMA (avoids deepcopy — weight_norm's non-leaf
        # hook tensors fail torch.Tensor.__deepcopy__). Stores detached
        # CPU-pinned clones keyed by parameter name; updated in-place on
        # each optimizer step.
        self._ema_enabled = bool(getattr(cfg, "use_ema", False))
        self._ema_state: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_ema(self):
        if not self._ema_enabled:
            return
        if self._ema_state is None:
            # Lazy build after first real forward (lazy-init Snake params
            # don't exist until then).
            self._ema_state = {
                name: p.detach().clone() for name, p in self.model.named_parameters()
            }
            return
        if self.global_step < int(getattr(self.cfg, "ema_warmup", 0)):
            for name, p in self.model.named_parameters():
                ep = self._ema_state.get(name)
                if ep is None or ep.shape != p.shape:
                    self._ema_state[name] = p.detach().clone()
                else:
                    ep.copy_(p.detach())
            return
        d = float(self.cfg.ema_decay)
        for name, p in self.model.named_parameters():
            ep = self._ema_state.get(name)
            if ep is None or ep.shape != p.shape:
                self._ema_state[name] = p.detach().clone()
            else:
                ep.mul_(d).add_(p.detach(), alpha=1.0 - d)

    def _swap_in_ema(self):
        """Swap live params with EMA; return undo callable."""
        if self._ema_state is None:
            return lambda: None
        backup: Dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            ep = self._ema_state.get(name)
            if ep is None:
                continue
            backup[name] = p.detach().clone()
            p.data.copy_(ep)

        def undo():
            for name, p in self.model.named_parameters():
                if name in backup:
                    p.data.copy_(backup[name])

        return undo

    # ------------------------------------------------------------------
    def _align(self, y: Tensor, y_hat: Tensor):
        T = min(y.shape[-1], y_hat.shape[-1])
        y = y[..., :T]
        y_hat = y_hat[..., :T]
        if y.ndim == 2:
            y = y.unsqueeze(1)
        if y_hat.ndim == 2:
            y_hat = y_hat.unsqueeze(1)
        return y, y_hat

    def _disc_inputs(self, y: Tensor, y_hat: Tensor):
        """Keep the discriminator in fp32 even when the generator uses bf16 AMP."""
        y, y_hat = self._align(y, y_hat)
        max_samples = int(getattr(self.cfg, "disc_max_samples", 0) or 0)
        if max_samples > 0 and y.shape[-1] > max_samples:
            max_start = y.shape[-1] - max_samples
            start = int(torch.randint(0, max_start + 1, (1,), device=y.device).item())
            y = y[..., start : start + max_samples]
            y_hat = y_hat[..., start : start + max_samples]
        return y.float(), y_hat.float()

    # ------------------------------------------------------------------
    def train_one_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.train()
        self.discriminator.train()

        # LR schedule (G only; D uses its own fixed lr)
        lr = _warmup_cosine_lr(
            self.global_step, self.cfg.lr, self.cfg.warmup_steps, self.cfg.max_steps
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        use_adv = self.global_step >= int(self.cfg.adv_start_step)

        # --------------------------------------------------------------
        # Generator step
        # --------------------------------------------------------------
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=(self.amp_dtype != torch.float32),
        ):
            enc = self._forward_model(batch)
            losses = self._compute_losses(enc)

        g_total = losses["loss/total"]

        g_adv = g_total.new_zeros(())
        g_fm = g_total.new_zeros(())
        if use_adv:
            y_real, y_fake = self._disc_inputs(enc["__audio"], enc["x_hat"])
            # No detach on y_fake here: gradients flow back into G.
            fake_out = self.discriminator(y_fake)
            with torch.no_grad():
                real_out_nograd = self.discriminator(y_real)
            g_adv = hinge_g_loss(fake_out)
            g_fm = feature_matching_loss(real_out_nograd, fake_out)
            g_total = g_total + self.cfg.w_adv * g_adv + self.cfg.w_feat_match * g_fm

        g_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.grad_clip
        )
        self.optimizer.step()
        self._update_ema()

        # --------------------------------------------------------------
        # Discriminator LR schedule (match generator decay)
        if getattr(self.cfg, "disc_lr_decay", False):
            disc_lr = _warmup_cosine_lr(
                self.global_step, self.cfg.disc_lr,
                self.cfg.warmup_steps, self.cfg.max_steps
            )
            for pg in self.opt_d.param_groups:
                pg["lr"] = disc_lr

        # Discriminator step
        # --------------------------------------------------------------
        d_loss = g_total.new_zeros(())
        if use_adv:
            self.opt_d.zero_grad(set_to_none=True)
            y_real, y_fake = self._disc_inputs(enc["__audio"].detach(), enc["x_hat"].detach())
            real_out = self.discriminator(y_real)
            fake_out = self.discriminator(y_fake)
            d_loss = hinge_d_loss(real_out, fake_out)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=self.cfg.grad_clip
            )
            self.opt_d.step()

        # --------------------------------------------------------------
        # Stats
        # --------------------------------------------------------------
        stats = {k: float(v.item()) if isinstance(v, Tensor) else float(v) for k, v in losses.items()}
        stats["loss/total"] = float(g_total.detach().item())
        stats["loss/g_adv"] = float(g_adv.detach().item()) if isinstance(g_adv, Tensor) else float(g_adv)
        stats["loss/g_fm"] = float(g_fm.detach().item()) if isinstance(g_fm, Tensor) else float(g_fm)
        stats["loss/d"] = float(d_loss.detach().item()) if isinstance(d_loss, Tensor) else float(d_loss)
        stats["adv/active"] = float(use_adv)
        stats["lr"] = lr
        stats["grad_norm/total"] = float(grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm)
        stats.update(self._latent_stats(enc))
        return stats

    # ------------------------------------------------------------------
    # EMA-aware validate + checkpoint (override parent)
    # ------------------------------------------------------------------
    def validate(self):
        if self._ema_state is None:
            return super().validate()
        undo = self._swap_in_ema()
        try:
            return super().validate()
        finally:
            undo()

    def _save_checkpoint(self, tag: str):
        path = super()._save_checkpoint(tag)
        # Save discriminator + opt_d 
        adv_path = path.with_name(path.stem + "_adv.pt")
        torch.save(
            {
                "discriminator": self.discriminator.state_dict(),
                "opt_d": self.opt_d.state_dict(),
            },
            adv_path,
        )
        # Save EMA state
        if self._ema_state is not None:
            ema_path = path.with_name(path.stem + "_ema.pt")
            torch.save(
                {"step": self.global_step, "model_ema": self._ema_state},
                ema_path,
            )
        return path

    def load_checkpoint(self, path):
        """Resume training from a checkpoint — restores model, optimizer,
        discriminator, opt_d, and EMA state."""
        from pathlib import Path as P
        path = P(path)
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("step", 0)
        self.best_val_recon = state.get("best_val_recon", float("inf"))
        # Restore anchor heads
        for name, sd in state.get("trainer_aux_heads", {}).items():
            head = getattr(self, name, None)
            if head is not None:
                head.load_state_dict(sd)
        # Restore discriminator
        adv_path = path.with_name(path.stem + "_adv.pt")
        if adv_path.exists():
            adv_state = torch.load(adv_path, map_location=self.device)
            self.discriminator.load_state_dict(adv_state["discriminator"])
            self.opt_d.load_state_dict(adv_state["opt_d"])
            print(f"[adv_trainer] restored disc + opt_d from {adv_path}")
        # Restore EMA
        ema_path = path.with_name(path.stem + "_ema.pt")
        if ema_path.exists() and self._ema_enabled:
            ema_state = torch.load(ema_path, map_location=self.device)
            self._ema_state = ema_state.get("model_ema")
            print(f"[adv_trainer] restored EMA from {ema_path}")
        print(f"[adv_trainer] resumed from step {self.global_step}")
