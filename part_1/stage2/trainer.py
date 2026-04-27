from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from stage1.trainer import Stage1Trainer, TrainerConfig, _warmup_cosine_lr
from .ay_loss_v2 import AnalyticYieldingLossV2, AYConfigV2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Stage2TrainerConfig(TrainerConfig):
    # Curriculum 
    curriculum_inv_end: int = 3000
    curriculum_equiv_end: int = 8000
    curriculum_comm_end: int = 15000

    # L_AY base weight (multiplies c_* * family_weight in AYConfigV2)
    w_ay: float = 1.0

    init_ckpt: Optional[Path] = None
    load_init_strict: bool = False

    deterministic_anchor: bool = True

    ay_cfg_v2: Optional[AYConfigV2] = None


# ---------------------------------------------------------------------------
# Curriculum function
# ---------------------------------------------------------------------------
def _linear_ramp(step: int, end: int) -> float:
    if end <= 0:
        return 1.0
    return float(min(max(step / end, 0.0), 1.0))


def compute_curriculum(step: int, cfg: Stage2TrainerConfig) -> Tuple[float, float, float]:
    c_inv = _linear_ramp(step, cfg.curriculum_inv_end)
    c_equiv = _linear_ramp(step - cfg.curriculum_inv_end, cfg.curriculum_equiv_end - cfg.curriculum_inv_end)
    c_comm = _linear_ramp(step - cfg.curriculum_equiv_end, cfg.curriculum_comm_end - cfg.curriculum_equiv_end)
    return c_inv, c_equiv, c_comm


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Stage2Trainer(Stage1Trainer):

    def __init__(
        self,
        cfg: Stage2TrainerConfig,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        # Feature-mode models (Variant A) are now supported — L_AY transforms
        # operate on aux features while feat_mel stays fixed as the content
        # encoder's invariant input.
        super().__init__(cfg, model, train_loader, val_loader)
        self.cfg: Stage2TrainerConfig = cfg

        ay_cfg = cfg.ay_cfg_v2 or AYConfigV2()
        self.ay = AnalyticYieldingLossV2(ay_cfg).to(self.device)

        # Initialise from Stage 1 checkpoint
        if cfg.init_ckpt is not None:
            self._load_init_ckpt(Path(cfg.init_ckpt))

    def _load_init_ckpt(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Stage 2 init_ckpt not found: {path}")
        state = torch.load(path, map_location=self.device)
        strict = bool(self.cfg.load_init_strict)
        missing, unexpected = self.model.load_state_dict(state["model"], strict=strict)
        print(f"[stage2] initialised model from {path} (step={state.get('step', '?')}) strict={strict}")
        if missing:
            print(f"[stage2] missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[stage2] unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

    def train_one_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.train()

        lr = _warmup_cosine_lr(
            self.global_step, self.cfg.lr, self.cfg.warmup_steps, self.cfg.max_steps
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)

        c_inv, c_equiv, c_comm = compute_curriculum(self.global_step, self.cfg)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=(self.amp_dtype != torch.float32),
        ):
            enc = self._forward_model(batch)
            stage1_losses = self._compute_losses(enc)

            if self.cfg.deterministic_anchor and self.is_variant_b:
                from transforms import build_aux_bundle
                aux_bundle = build_aux_bundle(
                    batch,
                    device=self.device,
                    is_variant_b=True,
                    aux_zero_mask=getattr(self.model.cfg, "aux_zero_mask", ()),
                )
                anchor = self.model.encode_from_codec_latent(
                    enc["feat"], *aux_bundle.encode_args(), deterministic=True
                )
                anchor["__audio"] = enc["__audio"]
                for k in ("__aux_pitch", "__aux_rhythm", "__aux_timbre"):
                    if k in enc:
                        anchor[k] = enc[k]
            else:
                # Feature-mode models (Variant A) and non-deterministic B:
                # use the training forward's output as anchor directly.
                anchor = enc

            audio = anchor["__audio"]
            ay_losses = self.ay(
                self.model,
                anchor,
                audio,
                batch,
                curriculum=(c_inv, c_equiv, c_comm),
            )

            total = stage1_losses["loss/total"] + self.cfg.w_ay * ay_losses["ay/total"]

        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.grad_clip
        )
        self.optimizer.step()

        # Merge stats
        stats = {
            k: float(v.item()) if isinstance(v, Tensor) else float(v)
            for k, v in stage1_losses.items()
        }
        stats["loss/total"] = float(total.item())
        stats["loss/stage1_total"] = float(stage1_losses["loss/total"].item())
        for k, v in ay_losses.items():
            stats[k] = float(v.item()) if isinstance(v, Tensor) else float(v)
        stats["lr"] = lr
        stats["grad_norm/total"] = float(grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm)
        stats.update(self._latent_stats(enc))
        return stats

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        sums: Dict[str, float] = {}
        n = 0
        for batch in self.val_loader:
            c_inv, c_equiv, c_comm = compute_curriculum(self.global_step, self.cfg)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=(self.amp_dtype != torch.float32),
            ):
                enc = self._forward_model(batch)
                stage1_losses = self._compute_losses(enc)
                ay_losses = self.ay(
                    self.model,
                    enc,
                    enc["__audio"],
                    batch,
                    curriculum=(c_inv, c_equiv, c_comm),
                )
                total = stage1_losses["loss/total"] + self.cfg.w_ay * ay_losses["ay/total"]
            n += 1
            sums["val/loss/total"] = sums.get("val/loss/total", 0.0) + float(total.item())
            for k, v in stage1_losses.items():
                val = float(v.item()) if isinstance(v, Tensor) else float(v)
                sums[f"val/{k}"] = sums.get(f"val/{k}", 0.0) + val
            for k, v in ay_losses.items():
                val = float(v.item()) if isinstance(v, Tensor) else float(v)
                sums[f"val/{k}"] = sums.get(f"val/{k}", 0.0) + val
        return {k: v / max(n, 1) for k, v in sums.items()}

    def fit(self):
        print(f"[stage2] device={self.device} amp_dtype={self.amp_dtype} steps={self.cfg.max_steps}")
        print(f"[stage2] model={type(self.model).__name__} variant_b={self.is_variant_b}")
        print(f"[stage2] out_dir={self.out_dir}")

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
                    f"stage1={stats['loss/stage1_total']:.4f} "
                    f"ay={stats['ay/total']:.4f} "
                    f"lr={stats['lr']:.2e}"
                )
                self._log(stats)

            if self.val_loader is not None and self.global_step % self.cfg.val_every == 0:
                val_stats = self.validate()
                self._log(val_stats)
                val_total = val_stats.get("val/loss/total", float("inf"))
                if val_total < self.best_val_recon:
                    self.best_val_recon = val_total
                    self._save_checkpoint("best")

            if self.global_step % self.cfg.ckpt_every == 0:
                self._save_checkpoint(f"step{self.global_step:08d}")

        self._save_checkpoint(f"step{self.global_step:08d}")
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
