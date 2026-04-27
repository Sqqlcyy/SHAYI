"""Stage 3 trainer — AYI probe on frozen AE latents.

Stage 3 reuses the same transform + auxiliary-sync path as Stage 2, but the
training target is now analytic delta in physical units rather than latent
delta. The AYI probe never edits latents during training.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from stage1.trainer import _warmup_cosine_lr
from transforms import (
    FeatureTransformConfig,
    FeatureTransformFamily,
    build_aux_bundle,
)

from .ayi_mlp import AYIMLP


@dataclass
class Stage3TrainerConfig:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 20_000
    grad_clip: float = 1.0

    pitch_min_st: float = -4.0
    pitch_max_st: float = 4.0
    rhythm_min_scale: float = 0.9
    rhythm_max_scale: float = 1.1111
    timbre_min_tilt_db: float = -6.0
    timbre_max_tilt_db: float = 6.0
    transform_backend: str = "feature"
    aux_sync_policy: str = "closed_form"
    pitch_bins_per_semitone: int = 7
    feature_fps: float = 86.1328125
    vibrato_min_amount: float = 0.5
    vibrato_max_amount: float = 2.0
    vibrato_lfo_hz: float = 5.0
    rhythm_shift_min_steps: int = -16
    rhythm_shift_max_steps: int = 16
    timbre_high_cut_min_hz: float = 3000.0
    timbre_high_cut_max_hz: float = 12000.0
    timbre_formant_min_scale: float = 0.85
    timbre_formant_max_scale: float = 1.15
    mel_fmin_hz: float = 0.0
    mel_fmax_hz: float = 22050.0
    pitch_probe_op: str = "pitch_shift"
    rhythm_probe_op: str = "rhythm_stretch"
    timbre_probe_op: str = "timbre_eq_tilt"

    w_pitch: float = 1.0
    w_rhythm: float = 1.0
    w_timbre: float = 1.0

    log_every: int = 50
    val_every: int = 1000
    ckpt_every: int = 2000
    out_dir: Path = Path("./runs/stage3")
    keep_last: int = 3

    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"

    encoder_ckpt: Optional[Path] = None
    device: str = "cuda"
    amp: bool = True
    seed: int = 0


class Stage3Trainer:
    def __init__(
        self,
        cfg: Stage3TrainerConfig,
        frozen_ae: nn.Module,
        ayi: AYIMLP,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.cfg = cfg
        self.backend = cfg.transform_backend.lower()
        if self.backend not in {"feature", "analytic_feature", "features"}:
            raise ValueError(
                f"Unsupported transform_backend={cfg.transform_backend}; "
                "Stage 3 now only supports feature-domain transforms."
            )
        if cfg.aux_sync_policy != "closed_form":
            raise ValueError(f"Unsupported aux_sync_policy={cfg.aux_sync_policy}")
        self.ae = frozen_ae
        self.ayi = ayi
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.ae.to(self.device)
        self.ayi.to(self.device)

        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        self.is_variant_b = bool(getattr(self.ae, "uses_aux_adapters", False))
        if getattr(self.ae, "input_mode", "waveform") == "feature":
            raise NotImplementedError(
                "Stage 3 probe for feature-domain Variant A needs the analytic-feature "
                "pair sampler. The current Stage 3 path is wired for B_CODEC's "
                "codec-latent trunk; run it with Variant B-Codec or add the "
                "feature-input probe path."
            )
        self.feature_family = FeatureTransformFamily(
            FeatureTransformConfig(
                pitch_min_st=cfg.pitch_min_st,
                pitch_max_st=cfg.pitch_max_st,
                pitch_bins_per_semitone=cfg.pitch_bins_per_semitone,
                vibrato_min_amount=cfg.vibrato_min_amount,
                vibrato_max_amount=cfg.vibrato_max_amount,
                vibrato_lfo_hz=cfg.vibrato_lfo_hz,
                feature_fps=cfg.feature_fps,
                rhythm_min_scale=cfg.rhythm_min_scale,
                rhythm_max_scale=cfg.rhythm_max_scale,
                rhythm_shift_min_steps=cfg.rhythm_shift_min_steps,
                rhythm_shift_max_steps=cfg.rhythm_shift_max_steps,
                timbre_min_tilt_db=cfg.timbre_min_tilt_db,
                timbre_max_tilt_db=cfg.timbre_max_tilt_db,
                timbre_high_cut_min_hz=cfg.timbre_high_cut_min_hz,
                timbre_high_cut_max_hz=cfg.timbre_high_cut_max_hz,
                timbre_formant_min_scale=cfg.timbre_formant_min_scale,
                timbre_formant_max_scale=cfg.timbre_formant_max_scale,
                mel_fmin_hz=cfg.mel_fmin_hz,
                mel_fmax_hz=cfg.mel_fmax_hz,
            )
        )

        self.optimizer = torch.optim.AdamW(
            self.ayi.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        self.amp_dtype = torch.bfloat16 if cfg.amp and self.device.type == "cuda" else torch.float32
        self.global_step = 0
        self.best_val = float("inf")
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._wandb = None
        self._init_wandb()

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
                config={k: (str(v) if isinstance(v, Path) else v) for k, v in self.cfg.__dict__.items()},
                dir=str(self.out_dir),
            )
            self._wandb = wandb
        except Exception as e:  # noqa: BLE001
            print(f"[stage3] WandB init failed ({e}); logging locally only.")

    def _encode(self, x: Tensor, aux_bundle) -> Dict[str, Tensor]:
        if self.is_variant_b:
            return self.ae.encode(x, *aux_bundle.encode_args())
        return self.ae.encode(x)

    def _encode_from_codec_latent(self, x: Tensor, codec_latent: Tensor, aux_bundle) -> Dict[str, Tensor]:
        if hasattr(self.ae, "encode_from_codec_latent"):
            return self.ae.encode_from_codec_latent(codec_latent, *aux_bundle.encode_args())
        return self._encode(x, aux_bundle)

    def _make_probe_batch(self, batch: Dict) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, float]]:
        x = batch["audio"].to(self.device, non_blocking=True)
        aux_anchor = build_aux_bundle(
            batch,
            device=self.device,
            is_variant_b=self.is_variant_b,
            aux_zero_mask=getattr(getattr(self.ae, "cfg", None), "aux_zero_mask", ()),
        )
        with torch.no_grad():
            enc_anchor = self._encode(x, aux_anchor)
            return self._make_feature_probe_batch(x, aux_anchor, enc_anchor)

    def _make_feature_probe_batch(
        self,
        x: Tensor,
        aux_anchor,
        enc_anchor: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, float]]:
        codec_latent = enc_anchor.get("feat")
        if codec_latent is None:
            raise RuntimeError("Feature-domain Stage 3 needs anchor['feat'] from the frozen B_CODEC encoder.")

        spec_p = self.feature_family.sample_spec(self.cfg.pitch_probe_op, device=self.device)
        spec_r = self.feature_family.sample_spec(self.cfg.rhythm_probe_op, device=self.device)
        spec_t = self.feature_family.sample_spec(self.cfg.timbre_probe_op, device=self.device)

        aux_p = self.feature_family.apply(aux_anchor, spec_p)
        aux_r = self.feature_family.apply(aux_anchor, spec_r)
        aux_t = self.feature_family.apply(aux_anchor, spec_t)

        enc_p = self._encode_from_codec_latent(x, codec_latent, aux_p)
        enc_r = self._encode_from_codec_latent(x, codec_latent, aux_r)
        enc_t = self._encode_from_codec_latent(x, codec_latent, aux_t)

        features = {
            "pitch": AYIMLP.build_features(enc_anchor["z_p"], enc_p["z_p"]),
            "rhythm": AYIMLP.build_features(enc_anchor["z_r"], enc_r["z_r"]),
            "timbre": AYIMLP.build_features(enc_anchor["z_t"], enc_t["z_t"]),
        }
        targets = {
            "pitch": torch.full((x.shape[0], 1), float(spec_p.value), device=self.device, dtype=x.dtype),
            "rhythm": torch.full((x.shape[0], 1), math.log(float(spec_r.value)), device=self.device, dtype=x.dtype),
            "timbre": torch.full((x.shape[0], 1), float(spec_t.value), device=self.device, dtype=x.dtype),
        }
        meta = {
            "a/pitch_st": float(spec_p.value),
            "a/rhythm_scale": float(spec_r.value),
            "a/timbre_db": float(spec_t.value),
        }
        return features, targets, meta

    def _step(self, batch: Dict, *, training: bool) -> Dict[str, float]:
        if training:
            self.ayi.train()
            lr = _warmup_cosine_lr(self.global_step, self.cfg.lr, self.cfg.warmup_steps, self.cfg.max_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.ayi.eval()
            lr = 0.0

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=(self.amp_dtype != torch.float32),
        ):
            features, targets, meta = self._make_probe_batch(batch)
            preds = self.ayi(
                phi_pitch=features["pitch"],
                phi_rhythm=features["rhythm"],
                phi_timbre=features["timbre"],
            )
            L_p = F.l1_loss(preds["pitch"], targets["pitch"])
            L_r = F.l1_loss(preds["rhythm"], targets["rhythm"])
            L_t = F.l1_loss(preds["timbre"], targets["timbre"])
            total = self.cfg.w_pitch * L_p + self.cfg.w_rhythm * L_r + self.cfg.w_timbre * L_t

        grad_norm = 0.0
        if training:
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.ayi.parameters(), max_norm=self.cfg.grad_clip)
            self.optimizer.step()

        with torch.no_grad():
            mae_p = float((preds["pitch"] - targets["pitch"]).abs().mean().item())
            mae_r = float((preds["rhythm"] - targets["rhythm"]).abs().mean().item())
            mae_t = float((preds["timbre"] - targets["timbre"]).abs().mean().item())

        stats = {
            "loss/total": float(total.item()),
            "loss/pitch": float(L_p.item()),
            "loss/rhythm": float(L_r.item()),
            "loss/timbre": float(L_t.item()),
            "mae/pitch": mae_p,
            "mae/rhythm": mae_r,
            "mae/timbre": mae_t,
            "feature/pitch_norm": float(features["pitch"].norm(dim=-1).mean().item()),
            "feature/rhythm_norm": float(features["rhythm"].norm(dim=-1).mean().item()),
            "feature/timbre_norm": float(features["timbre"].norm(dim=-1).mean().item()),
            "lr": lr,
            "grad_norm/total": float(grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm),
            **meta,
        }
        return stats

    def train_one_step(self, batch: Dict) -> Dict[str, float]:
        return self._step(batch, training=True)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        sums: Dict[str, float] = {}
        n = 0
        for batch in self.val_loader:
            stats = self._step(batch, training=False)
            n += 1
            for k, v in stats.items():
                if k in {"lr", "grad_norm/total"}:
                    continue
                sums[f"val/{k}"] = sums.get(f"val/{k}", 0.0) + float(v)
        return {k: v / max(n, 1) for k, v in sums.items()}

    def _log(self, stats: Dict[str, float]):
        stats = {"step": self.global_step, **stats}
        if self._wandb is not None:
            self._wandb.log(stats, step=self.global_step)
        with open(self.out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")

    def _save_checkpoint(self, tag: str):
        path = self.out_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "step": self.global_step,
                "ayi": self.ayi.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_val": self.best_val,
                "cfg": {k: (str(v) if isinstance(v, Path) else v) for k, v in self.cfg.__dict__.items()},
            },
            path,
        )
        ckpts = sorted(self.out_dir.glob("ckpt_step*.pt"))
        if len(ckpts) > self.cfg.keep_last:
            for p in ckpts[: -self.cfg.keep_last]:
                try:
                    p.unlink()
                except OSError:
                    pass

    def fit(self):
        print(f"[stage3] device={self.device} amp_dtype={self.amp_dtype} steps={self.cfg.max_steps}")
        print(f"[stage3] AE type={type(self.ae).__name__} variant_b={self.is_variant_b}")
        print(f"[stage3] out_dir={self.out_dir}")

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
                    f"total={stats['loss/total']:.4f} "
                    f"p={stats['loss/pitch']:.4f} r={stats['loss/rhythm']:.4f} t={stats['loss/timbre']:.4f} "
                    f"maep={stats['mae/pitch']:.3f} maer={stats['mae/rhythm']:.3f} maet={stats['mae/timbre']:.3f}"
                )
                self._log(stats)

            if self.val_loader is not None and self.global_step % self.cfg.val_every == 0:
                val_stats = self.validate()
                self._log(val_stats)
                val_total = val_stats.get("val/loss/total", float("inf"))
                if val_total < self.best_val:
                    self.best_val = val_total
                    self._save_checkpoint("best")

            if self.global_step % self.cfg.ckpt_every == 0:
                self._save_checkpoint(f"step{self.global_step:08d}")

        self._save_checkpoint(f"step{self.global_step:08d}")
        if self._wandb is not None:
            self._wandb.finish()
