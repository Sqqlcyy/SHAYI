from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import Stage1AEDataset, Stage1DataConfig, stage1_collate  # noqa: E402
from stage2.ay_loss_v2 import AYConfigV2  # noqa: E402
from stage2.trainer import Stage2Trainer, Stage2TrainerConfig  # noqa: E402
# Model is now built via runtime.build_model (supports all variants).


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(model_cfg: dict):
    from runtime import build_model
    return build_model(model_cfg)


def _build_dataset(data_cfg: dict, split: str) -> Stage1AEDataset:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--init_ckpt", type=Path, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--wandb_mode", type=str, default=None)
    args = ap.parse_args()

    raw = _load_yaml(args.config)
    from stage1.run import _validate_cfg_for_variant  # reuse Stage-1 variant guard
    _validate_cfg_for_variant(raw)

    seed = raw.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_ds = _build_dataset(raw["data"], "train")
    val_ds = None
    if raw["data"].get("val_list"):
        try:
            val_ds = _build_dataset(raw["data"], "val")
        except RuntimeError as e:
            print(f"[warn] validation dataset could not be built: {e}")

    batch_size = int(raw["train"].get("batch_size", 8))
    num_workers = int(raw["train"].get("num_workers", 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=stage1_collate,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            collate_fn=stage1_collate,
            drop_last=False,
            pin_memory=True,
        )

    model = _build_model(raw["model"])

    tr_raw = raw["train"]
    ay_raw = raw.get("ay", {})

    # All transform-family / weight fields read from yaml's ``ay:`` block
    # and passed to AYConfigV2 (v1 was deleted; ratio losses are mainline).
    _ay_shared = dict(
        pitch_min_st=ay_raw.get("pitch_min_st", -4.0),
        pitch_max_st=ay_raw.get("pitch_max_st", 4.0),
        rhythm_min_scale=ay_raw.get("rhythm_min_scale", 0.9),
        rhythm_max_scale=ay_raw.get("rhythm_max_scale", 1.1111),
        timbre_min_tilt_db=ay_raw.get("timbre_min_tilt_db", -6.0),
        timbre_max_tilt_db=ay_raw.get("timbre_max_tilt_db", 6.0),
        transform_backend=ay_raw.get("transform_backend", "feature"),
        aux_sync_policy=ay_raw.get("aux_sync_policy", "closed_form"),
        feature_transform_ops=tuple(ay_raw.get("feature_transform_ops", (
            "pitch_shift",
            "pitch_vibrato",
            "rhythm_stretch",
            "rhythm_shift",
            "timbre_high_cut",
            "timbre_eq_tilt",
            "timbre_formant",
        ))),
        pitch_bins_per_semitone=ay_raw.get("pitch_bins_per_semitone", 7),
        vibrato_min_amount=ay_raw.get("vibrato_min_amount", 0.5),
        vibrato_max_amount=ay_raw.get("vibrato_max_amount", 2.0),
        vibrato_lfo_hz=ay_raw.get("vibrato_lfo_hz", 5.0),
        feature_fps=ay_raw.get("feature_fps", raw["data"].get("sample_rate", 44100) / raw["data"].get("feat_hop", 512)),
        rhythm_shift_min_steps=ay_raw.get("rhythm_shift_min_steps", -16),
        rhythm_shift_max_steps=ay_raw.get("rhythm_shift_max_steps", 16),
        timbre_high_cut_min_hz=ay_raw.get("timbre_high_cut_min_hz", 3000.0),
        timbre_high_cut_max_hz=ay_raw.get("timbre_high_cut_max_hz", 12000.0),
        timbre_formant_min_scale=ay_raw.get("timbre_formant_min_scale", 0.85),
        timbre_formant_max_scale=ay_raw.get("timbre_formant_max_scale", 1.15),
        mel_fmin_hz=ay_raw.get("mel_fmin_hz", 0.0),
        mel_fmax_hz=ay_raw.get("mel_fmax_hz", 22050.0),
        w_equiv=ay_raw.get("w_equiv", 1.0),
        w_inv=ay_raw.get("w_inv", 1.0),
        w_comm=ay_raw.get("w_comm", 0.5),
        include_rhythm_equivariance=ay_raw.get("include_rhythm_equivariance", True),
        include_commutativity=ay_raw.get("include_commutativity", True),
    )

    ay_cfg_v2 = AYConfigV2(
        **_ay_shared,
        pesto_n_pitch_bins=ay_raw.get(
            "pesto_n_pitch_bins", raw.get("model", {}).get("n_pitch_bins", 128)
        ),
        pesto_bins_per_semitone=ay_raw.get("pesto_bins_per_semitone", 1),
        scalar_tau=ay_raw.get("scalar_tau", 1.0),
        timbre_gamma_init=ay_raw.get("timbre_gamma_init", 1.0),
    )

    tcfg = Stage2TrainerConfig(
        lr=tr_raw.get("lr", 1.5e-4),   # lower lr than Stage 1 for finetuning
        betas=tuple(tr_raw.get("betas", (0.9, 0.99))),
        weight_decay=tr_raw.get("weight_decay", 1e-4),
        grad_clip=tr_raw.get("grad_clip", 1.0),
        accum_steps=tr_raw.get("accum_steps", 1),
        warmup_steps=tr_raw.get("warmup_steps", 500),
        max_steps=args.max_steps or tr_raw.get("max_steps", 30_000),
        w_recon=tr_raw.get("w_recon", 1.0),
        w_ortho=tr_raw.get("w_ortho", 0.1),
        ortho_form=tr_raw.get("ortho_form", "cross_cov"),
        w_kl=tr_raw.get("w_kl", 1e-4),
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
        require_codec_distill=tr_raw.get("require_codec_distill", True),
        load_init_strict=tr_raw.get("load_init_strict", False),
        deterministic_anchor=tr_raw.get("deterministic_anchor", True),
        w_moe_t=tr_raw.get("w_moe_t", tr_raw.get("w_moe", 0.0)),
        w_moe_r=tr_raw.get("w_moe_r", 0.0),
        moe_k=tr_raw.get("moe_k", 8),
        w_codec_distill=tr_raw.get("w_codec_distill", 1.0),
        sample_rate=raw["data"].get("sample_rate", 32000),
        fft_sizes=tuple(tr_raw.get("fft_sizes", (1024, 2048, 4096))),
        hop_sizes=tuple(tr_raw.get("hop_sizes", (120, 240, 480))),
        win_lengths=tuple(tr_raw.get("win_lengths", (960, 1920, 3840))),
        n_mels=tr_raw.get("n_mels", 128),
        w_logmel=tr_raw.get("w_logmel", 0.5),
        log_every=tr_raw.get("log_every", 50),
        val_every=tr_raw.get("val_every", 1000),
        ckpt_every=tr_raw.get("ckpt_every", 2000),
        media_every=tr_raw.get("media_every", 1000),
        out_dir=Path(args.out_dir or tr_raw.get("out_dir", "./runs/stage2")),
        keep_last=tr_raw.get("keep_last", 3),
        wandb_project=tr_raw.get("wandb_project"),
        wandb_run_name=tr_raw.get("wandb_run_name"),
        wandb_entity=tr_raw.get("wandb_entity"),
        wandb_mode=args.wandb_mode or tr_raw.get("wandb_mode", "online"),
        tb_enabled=tr_raw.get("tb_enabled", True),
        tb_flush_every=tr_raw.get("tb_flush_every", 20),
        device=tr_raw.get("device", "cuda"),
        amp=tr_raw.get("amp", True),
        seed=seed,
        curriculum_inv_end=tr_raw.get("curriculum_inv_end", 3000),
        curriculum_equiv_end=tr_raw.get("curriculum_equiv_end", 8000),
        curriculum_comm_end=tr_raw.get("curriculum_comm_end", 15000),
        w_ay=tr_raw.get("w_ay", 1.0),
        init_ckpt=args.init_ckpt or (Path(tr_raw["init_ckpt"]) if tr_raw.get("init_ckpt") else None),
        ay_cfg_v2=ay_cfg_v2,
    )

    trainer = Stage2Trainer(tcfg, model, train_loader, val_loader)
    trainer.fit()


if __name__ == "__main__":
    main()
