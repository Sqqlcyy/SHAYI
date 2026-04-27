 #!/usr/bin/env python
"""
DDP wrapper for SHAYI Stage 1 AE training.

Usage (6-GPU):
  cd /root/autodl-tmp/SHAYI/part_1
  torchrun --standalone --nproc_per_node=6 stage1/run_ddp.py \
      --config configs/train_a_nocontent_full.yaml \
      --out_dir runs/stage1_nocontent_ddp

Single-GPU fallback (same as run.py):
  python stage1/run_ddp.py --config configs/train_a_nocontent_full.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import Stage1AEDataset, Stage1DataConfig, stage1_collate
from stage1.trainer import Stage1Trainer, TrainerConfig
from stage1.adv_trainer import Stage1AdvTrainer, AdvTrainerConfig


# =====================================================================
#  DDP setup
# =====================================================================

def setup_ddp():
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_ddp(is_ddp):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


# =====================================================================
#  Yaml / model / dataset builders (reused from run.py)
# =====================================================================

def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_model(model_cfg):
    from runtime import build_model
    return build_model(model_cfg)


def _build_dataset(data_cfg, split):
    list_path = data_cfg.get(f"{split}_list")
    cfg = Stage1DataConfig(
        roots=[Path(r) for r in data_cfg.get("roots", [])],
        track_list=Path(list_path) if list_path else None,
        sample_rate=data_cfg.get("sample_rate", 44100),
        crop_seconds=data_cfg.get("crop_seconds", 10.0),
        feat_hop=data_cfg.get("feat_hop", 512),
        latent_hop=data_cfg.get("latent_hop", 512),
        normalize_audio=data_cfg.get("normalize_audio", True),
        augment=data_cfg.get("augment", False) and split == "train",
        pitch_feat_name=data_cfg.get("pitch_feat_name", "pitch_salience_instru_nondrum.npy"),
        rhythm_feat_name=data_cfg.get("rhythm_feat_name", "rhythm_instru.npy"),
        rhythm_multi_feat_name=data_cfg.get("rhythm_multi_feat_name", "rhythm_multi_instru.npy"),
        timbre_feat_name=data_cfg.get("timbre_feat_name", "mfcc_instru.npy"),
        envelope_feat_name=data_cfg.get("envelope_feat_name", "envelope_instru.npy"),
        prefer_envelope_timbre=data_cfg.get("prefer_envelope_timbre", False),
        spec_feat_name=data_cfg.get("spec_feat_name", "spec_instru.npy"),
        mel_linear_feat_name=data_cfg.get("mel_linear_feat_name", "mel_linear_instru.npy"),
        n_pitch_bins=data_cfg.get("n_pitch_bins", 588),
        n_mels=data_cfg.get("n_mels", 128),
        n_mfcc=data_cfg.get("n_mfcc", 20),
        n_rhythm_channels=data_cfg.get("n_rhythm_channels", 1),
        n_envelope_dim=data_cfg.get("n_envelope_dim", 80),
        allow_missing_aux=data_cfg.get("allow_missing_aux", False),
    )
    return Stage1AEDataset(cfg)


# =====================================================================
#  Monkey-patch trainer for DDP
# =====================================================================

def _patch_trainer_for_ddp(trainer, rank, world_size, local_rank):
    """
    Minimally invasive DDP patch:
    1. Wrap model in DDP
    2. Override _save_checkpoint to only save on rank 0
    3. Override _log to only log on rank 0
    4. Store rank info
    """
    trainer._ddp_rank = rank
    trainer._ddp_world_size = world_size
    trainer._ddp_local_rank = local_rank
    trainer._ddp_is_main = (rank == 0)

    # Device override
    device = torch.device(f"cuda:{local_rank}")
    trainer.device = device
    trainer.model.to(device)
    trainer.recon_loss.to(device)
    trainer.ortho_loss.to(device)

    # Wrap model
    trainer.model = DDP(
        trainer.model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,  # adversary heads may not fire every step
    )

    # Patch: only rank 0 saves checkpoints
    _orig_save = trainer._save_checkpoint
    def _save_checkpoint_ddp(tag):
        if trainer._ddp_is_main:
            # DDP wrapper: unwrap model for saving
            orig_model = trainer.model
            trainer.model = orig_model.module
            path = _orig_save(tag)
            trainer.model = orig_model
            return path
        return None
    trainer._save_checkpoint = _save_checkpoint_ddp

    # Patch: only rank 0 logs
    _orig_log = trainer._log
    def _log_ddp(stats):
        if trainer._ddp_is_main:
            _orig_log(stats)
    trainer._log = _log_ddp

    # Patch: only rank 0 prints
    import builtins
    if not trainer._ddp_is_main:
        def _noop_print(*args, **kwargs):
            pass
        trainer._print = _noop_print
    
    # Patch _forward_model: need to handle model.module for attribute access
    _orig_forward = trainer._forward_model
    def _forward_model_ddp(batch, **kwargs):
        # Temporarily expose the raw model's attributes through DDP wrapper
        raw = trainer.model.module
        # The forward model checks getattr(model, ...) for various flags
        # DDP doesn't forward these, so we patch them
        for attr in ('uses_aux_adapters', 'input_mode', 'needs_target_wav',
                      'uses_envelope_aux', 'uses_adv_kwargs', 'cfg',
                      '_global_step', 'pitch_adversary',
                      'rhythm_scalar_proj', 'timbre_brightness_proj',
                      'trunk_p', 'trunk_r', 'trunk_t'):
            if hasattr(raw, attr) and not hasattr(trainer.model, attr):
                try:
                    setattr(trainer.model, attr, getattr(raw, attr))
                except (AttributeError, TypeError):
                    pass
        return _orig_forward(batch, **kwargs)
    trainer._forward_model = _forward_model_ddp

    return trainer


def _patch_adv_trainer_for_ddp(trainer, rank, world_size, local_rank):
    """Same as above but also wraps discriminator."""
    trainer = _patch_trainer_for_ddp(trainer, rank, world_size, local_rank)
    
    device = torch.device(f"cuda:{local_rank}")
    
    # Wrap discriminators if they exist
    if hasattr(trainer, 'mpd') and trainer.mpd is not None:
        trainer.mpd = trainer.mpd.to(device)
        trainer.mpd = DDP(trainer.mpd, device_ids=[local_rank],
                         find_unused_parameters=True)
    if hasattr(trainer, 'msd') and trainer.msd is not None:
        trainer.msd = trainer.msd.to(device)
        trainer.msd = DDP(trainer.msd, device_ids=[local_rank],
                         find_unused_parameters=True)
    if hasattr(trainer, 'mssd') and trainer.mssd is not None:
        trainer.mssd = trainer.mssd.to(device)
        trainer.mssd = DDP(trainer.mssd, device_ids=[local_rank],
                         find_unused_parameters=True)
    
    return trainer


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--wandb_mode", type=str, default="offline")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    is_ddp, rank, world_size, local_rank = setup_ddp()

    raw = _load_yaml(args.config)
    seed = raw.get("seed", 42)
    np.random.seed(seed + rank)  # different seed per rank for augmentation diversity
    torch.manual_seed(seed + rank)

    # ---- Data ----
    train_ds = _build_dataset(raw["data"], "train")
    val_ds = None
    if raw["data"].get("val_list"):
        try:
            val_ds = _build_dataset(raw["data"], "val")
        except Exception as e:
            if rank == 0:
                print(f"[warn] val dataset failed: {e}")

    batch_size = int(raw["train"].get("batch_size", 4))
    num_workers = int(raw["train"].get("num_workers", 4))

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=stage1_collate,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                          rank=rank, shuffle=False) if is_ddp else None
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=max(1, num_workers // 2),
            collate_fn=stage1_collate,
            drop_last=False,
            pin_memory=True,
        )

    # ---- Model ----
    model = _build_model(raw["model"])

    # ---- Trainer config ----
    tr_raw = raw["train"]
    use_adv = bool(tr_raw.get("use_adv", False))
    out_dir = args.out_dir or Path(tr_raw.get("out_dir", "./runs/stage1_nocontent"))
    max_steps = args.max_steps or tr_raw.get("max_steps", 15000)

    # Build trainer config (same as run.py)
    trainer_config_cls = AdvTrainerConfig if use_adv else TrainerConfig
    tcfg = trainer_config_cls(
        lr=tr_raw.get("lr", 3e-4),
        betas=tuple(tr_raw.get("betas", (0.9, 0.99))),
        weight_decay=tr_raw.get("weight_decay", 1e-4),
        grad_clip=tr_raw.get("grad_clip", 1.0),
        accum_steps=tr_raw.get("accum_steps", 1),
        warmup_steps=tr_raw.get("warmup_steps", 500),
        max_steps=max_steps,
        w_recon=tr_raw.get("w_recon", 1.0),
        w_ortho=tr_raw.get("w_ortho", 0.1),
        ortho_form=tr_raw.get("ortho_form", "cross_cov"),
        w_kl=tr_raw.get("w_kl", 0.0),
        kl_beta_p=tr_raw.get("kl_beta_p", 1.0),
        kl_beta_r=tr_raw.get("kl_beta_r", 1.0),
        kl_beta_t=tr_raw.get("kl_beta_t", 1.0),
        w_adv_pitch_on_t=tr_raw.get("w_adv_pitch_on_t", 0.0),
        leakage_probe_every=tr_raw.get("leakage_probe_every", 200),
        mi_probe_every=tr_raw.get("mi_probe_every", 2000),
        mi_inner_steps=tr_raw.get("mi_inner_steps", 200),
        mi_hidden=tr_raw.get("mi_hidden", 256),
        w_timbre_anchor=tr_raw.get("w_timbre_anchor", 0.0),
        timbre_anchor_warmup=tr_raw.get("timbre_anchor_warmup", 1500),
        timbre_anchor_decay_end=tr_raw.get("timbre_anchor_decay_end", 5000),
        w_content_adv=tr_raw.get("w_content_adv", 0.0),
        w_adv_pitch_on_zt=tr_raw.get("w_adv_pitch_on_zt", 0.0),
        w_adv_rhythm_on_zt=tr_raw.get("w_adv_rhythm_on_zt", 0.0),
        w_adv_pitch_on_zr=tr_raw.get("w_adv_pitch_on_zr", 0.0),
        w_adv_envelope_on_zr=tr_raw.get("w_adv_envelope_on_zr", 0.0),
        w_rhythm_anchor=tr_raw.get("w_rhythm_anchor", 0.0),
        rhythm_anchor_warmup=tr_raw.get("rhythm_anchor_warmup", 1500),
        rhythm_anchor_decay_end=tr_raw.get("rhythm_anchor_decay_end", 5000),
        require_codec_distill=tr_raw.get("require_codec_distill", False),
        w_moe_t=tr_raw.get("w_moe_t", 0.0),
        w_moe_r=tr_raw.get("w_moe_r", 0.0),
        moe_k=tr_raw.get("moe_k", 8),
        w_codec_distill=tr_raw.get("w_codec_distill", 0.0),
        w_trunk_l1=tr_raw.get("w_trunk_l1", 0.0),
        trunk_l1_form=tr_raw.get("trunk_l1_form", "group_lasso"),
        aux_dropout_p=tr_raw.get("aux_dropout_p", 0.0),
        w_pitch_anchor=tr_raw.get("w_pitch_anchor", 0.0),
        pitch_anchor_warmup=tr_raw.get("pitch_anchor_warmup", 1500),
        pitch_anchor_decay_end=tr_raw.get("pitch_anchor_decay_end", 5000),
        pitch_anchor_tau=tr_raw.get("pitch_anchor_tau", 0.5),
        sample_rate=raw["data"].get("sample_rate", 44100),
        fft_sizes=tuple(tr_raw.get("fft_sizes", (2048, 4096, 8192))),
        hop_sizes=tuple(tr_raw.get("hop_sizes", (256, 512, 1024))),
        win_lengths=tuple(tr_raw.get("win_lengths", (2048, 4096, 8192))),
        n_mels=tr_raw.get("n_mels", 256),
        w_logmel=tr_raw.get("w_logmel", 1.0),
        w_wav_l1=tr_raw.get("w_wav_l1", 1.0),
        log_every=tr_raw.get("log_every", 20),
        val_every=tr_raw.get("val_every", 500),
        ckpt_every=tr_raw.get("ckpt_every", 500),
        media_every=tr_raw.get("media_every", 1000),
        out_dir=out_dir,
        keep_last=tr_raw.get("keep_last", 20),
        wandb_project=tr_raw.get("wandb_project") if rank == 0 else None,
        wandb_run_name=tr_raw.get("wandb_run_name"),
        wandb_entity=tr_raw.get("wandb_entity"),
        wandb_mode=args.wandb_mode if rank == 0 else "disabled",
        tb_enabled=tr_raw.get("tb_enabled", True) and rank == 0,
        tb_flush_every=tr_raw.get("tb_flush_every", 20),
        device=f"cuda:{local_rank}" if is_ddp else tr_raw.get("device", "cuda"),
        amp=tr_raw.get("amp", True),
        seed=seed,
    )

    if use_adv:
        tcfg.w_adv = tr_raw.get("w_adv", 1.0)
        tcfg.w_feat_match = tr_raw.get("w_feat_match", 10.0)
        tcfg.disc_lr = tr_raw.get("disc_lr", 3e-4)
        tcfg.disc_betas = tuple(tr_raw.get("disc_betas", (0.8, 0.9)))
        tcfg.disc_weight_decay = tr_raw.get("disc_weight_decay", 0.0)
        tcfg.adv_start_step = int(tr_raw.get("adv_start_step", 1000))
        tcfg.disc_periods = tuple(tr_raw.get("disc_periods", (2, 3, 5, 7, 11)))
        tcfg.disc_n_scales = int(tr_raw.get("disc_n_scales", 3))
        tcfg.use_mssd = bool(tr_raw.get("use_mssd", False))
        tcfg.use_ema = bool(tr_raw.get("use_ema", False))
        tcfg.ema_decay = float(tr_raw.get("ema_decay", 0.999))
        tcfg.ema_warmup = int(tr_raw.get("ema_warmup", 1000))
        trainer = Stage1AdvTrainer(tcfg, model, train_loader, val_loader)
    else:
        trainer = Stage1Trainer(tcfg, model, train_loader, val_loader)

    # ---- DDP patch ----
    if is_ddp:
        if use_adv:
            trainer = _patch_adv_trainer_for_ddp(trainer, rank, world_size, local_rank)
        else:
            trainer = _patch_trainer_for_ddp(trainer, rank, world_size, local_rank)

    # ---- Resume ----
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=trainer.device)
        raw_model = trainer.model.module if is_ddp else trainer.model
        missing, unexpected = raw_model.load_state_dict(ck["model"], strict=False)
        if rank == 0:
            print(f"[resume] missing keys: {len(missing)}")
            print(f"[resume] unexpected keys: {len(unexpected)}")
            if len(unexpected) > 0:
                print("[resume] first unexpected:", unexpected[:10])
            if len(missing) > 0:
                print("[resume] first missing:", missing[:10])
        if "optimizer" in ck:
            trainer.optimizer.load_state_dict(ck["optimizer"])
        trainer.global_step = ck.get("step", 0)
        trainer.best_val_recon = ck.get("best_val_recon", float("inf"))
        if rank == 0:
            print(f"[resume] loaded from {args.resume} at step {trainer.global_step}")

    # ---- Sanity check (rank 0 only) ----
    if rank == 0:
        with torch.no_grad():
            sample = next(iter(train_loader))
            enc = trainer._forward_model(sample)
            print("[sanity] latent shapes:")
            for k in ("z_p", "z_r", "z_t", "x_hat"):
                if k in enc:
                    print(f"  {k}: {tuple(enc[k].shape)}")

    # ---- Patch fit() to set epoch on sampler ----
    if is_ddp and train_sampler is not None:
        _orig_fit = trainer.fit

        def _fit_ddp():
            print(f"[DDP rank={rank}] Starting training, world_size={world_size}")
            t0 = time.time()
            it = iter(train_loader)
            epoch = 0

            while trainer.global_step < trainer.cfg.max_steps:
                try:
                    batch = next(it)
                except StopIteration:
                    epoch += 1
                    train_sampler.set_epoch(epoch)
                    it = iter(train_loader)
                    batch = next(it)

                stats = trainer.train_one_step(batch)
                trainer.global_step += 1

                if trainer.global_step % trainer.cfg.log_every == 0:
                    dt = time.time() - t0
                    stats["time/sec_per_step"] = dt / max(trainer.cfg.log_every, 1)
                    t0 = time.time()
                    if rank == 0:
                        print(
                            f"step={trainer.global_step} "
                            f"loss={stats['loss/total']:.4f} "
                            f"recon={stats['loss/recon']:.4f} "
                            f"ortho={stats['loss/ortho']:.4e} "
                            f"lr={stats['lr']:.2e}"
                        )
                    trainer._log(stats)

                if val_loader is not None and trainer.global_step % trainer.cfg.val_every == 0:
                    val_stats = trainer.validate()
                    trainer._log(val_stats)
                    val_recon = val_stats.get("val/loss/recon", float("inf"))
                    if val_recon < trainer.best_val_recon:
                        trainer.best_val_recon = val_recon
                        trainer._save_checkpoint("best")
                    if is_ddp:
                        # Save EMA too if using AdvTrainer
                        if hasattr(trainer, '_ema_model') and trainer._ema_model is not None:
                            trainer._save_checkpoint("best_ema")

                if trainer.global_step % trainer.cfg.ckpt_every == 0:
                    trainer._save_checkpoint(f"step{trainer.global_step:08d}")

            # Final
            trainer._save_checkpoint(f"step{trainer.global_step:08d}")
            if rank == 0:
                print(f"[DDP] Training complete at step {trainer.global_step}")

        _fit_ddp()
    else:
        trainer.fit()

    cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
