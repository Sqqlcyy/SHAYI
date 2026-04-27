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
from stage3.ayi_mlp import AYIMLP, AYIMLPConfig  # noqa: E402
from stage3.trainer import Stage3Trainer, Stage3TrainerConfig  # noqa: E402
from ae_variant_b_codec import VariantBCodec, VariantBCodecConfig  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(model_cfg: dict):
    variant = model_cfg.get("variant", "B_CODEC").upper().replace("-", "_")
    if variant not in {"B_CODEC", "BCODEC", "CODEC"}:
        raise ValueError(f"Unsupported variant {variant!r}; expected B_CODEC.")
    return VariantBCodec(VariantBCodecConfig(**{k: v for k, v in model_cfg.items() if k != "variant"}))


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
        augment=False,   # never augment in Stage 3 — would contaminate the delta target
        pitch_feat_name=data_cfg.get("pitch_feat_name", "pitch_salience_instru_nondrum.npy"),
        rhythm_feat_name=data_cfg.get("rhythm_feat_name", "rhythm_instru.npy"),
        timbre_feat_name=data_cfg.get("timbre_feat_name", "mfcc_instru.npy"),
        spec_feat_name=data_cfg.get("spec_feat_name", "spec_instru.npy"),
        mel_linear_feat_name=data_cfg.get("mel_linear_feat_name", "mel_linear_instru.npy"),
        n_pitch_bins=data_cfg.get("n_pitch_bins", 588),
        n_mels=data_cfg.get("n_mels", 128),
        n_mfcc=data_cfg.get("n_mfcc", 20),
        allow_missing_aux=data_cfg.get("allow_missing_aux", False),
    )
    return Stage1AEDataset(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--encoder_ckpt", type=Path, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--wandb_mode", type=str, default=None)
    args = ap.parse_args()

    raw = _load_yaml(args.config)

    seed = raw.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data
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

    # AE (frozen)
    ae_cfg = raw["ae"]
    ae = _build_model(ae_cfg)
    encoder_ckpt = args.encoder_ckpt or (
        Path(raw["train"]["encoder_ckpt"]) if raw["train"].get("encoder_ckpt") else None
    )
    if encoder_ckpt is None:
        raise SystemExit(
            "Stage 3 requires a frozen encoder checkpoint — pass --encoder_ckpt or set train.encoder_ckpt in the YAML."
        )
    state = torch.load(encoder_ckpt, map_location="cpu")
    ae.load_state_dict(state["model"], strict=True)
    print(f"[stage3] loaded AE from {encoder_ckpt} (step={state.get('step', '?')})")

    # AYI MLP
    ayi_raw = raw.get("ayi", {})
    ayi_cfg = AYIMLPConfig(
        feature_dim=ayi_raw.get("feature_dim", 7),
        hidden_dim=ayi_raw.get("hidden_dim", 256),
        n_layers=ayi_raw.get("n_layers", 3),
        activation=ayi_raw.get("activation", "gelu"),
        zero_init_last=ayi_raw.get("zero_init_last", True),
    )
    ayi = AYIMLP(ayi_cfg)

    # Trainer config
    tr_raw = raw["train"]
    tcfg = Stage3TrainerConfig(
        lr=tr_raw.get("lr", 1e-3),
        betas=tuple(tr_raw.get("betas", (0.9, 0.999))),
        weight_decay=tr_raw.get("weight_decay", 1e-5),
        warmup_steps=tr_raw.get("warmup_steps", 500),
        max_steps=args.max_steps or tr_raw.get("max_steps", 20_000),
        grad_clip=tr_raw.get("grad_clip", 1.0),
        pitch_min_st=tr_raw.get("pitch_min_st", -4.0),
        pitch_max_st=tr_raw.get("pitch_max_st", 4.0),
        rhythm_min_scale=tr_raw.get("rhythm_min_scale", 0.9),
        rhythm_max_scale=tr_raw.get("rhythm_max_scale", 1.1111),
        timbre_min_tilt_db=tr_raw.get("timbre_min_tilt_db", -6.0),
        timbre_max_tilt_db=tr_raw.get("timbre_max_tilt_db", 6.0),
        transform_backend=tr_raw.get("transform_backend", "feature"),
        aux_sync_policy=tr_raw.get("aux_sync_policy", "closed_form"),
        pitch_bins_per_semitone=tr_raw.get("pitch_bins_per_semitone", 7),
        feature_fps=tr_raw.get("feature_fps", raw["data"].get("sample_rate", 44100) / raw["data"].get("feat_hop", 512)),
        vibrato_min_amount=tr_raw.get("vibrato_min_amount", 0.5),
        vibrato_max_amount=tr_raw.get("vibrato_max_amount", 2.0),
        vibrato_lfo_hz=tr_raw.get("vibrato_lfo_hz", 5.0),
        rhythm_shift_min_steps=tr_raw.get("rhythm_shift_min_steps", -16),
        rhythm_shift_max_steps=tr_raw.get("rhythm_shift_max_steps", 16),
        timbre_high_cut_min_hz=tr_raw.get("timbre_high_cut_min_hz", 3000.0),
        timbre_high_cut_max_hz=tr_raw.get("timbre_high_cut_max_hz", 12000.0),
        timbre_formant_min_scale=tr_raw.get("timbre_formant_min_scale", 0.85),
        timbre_formant_max_scale=tr_raw.get("timbre_formant_max_scale", 1.15),
        mel_fmin_hz=tr_raw.get("mel_fmin_hz", 0.0),
        mel_fmax_hz=tr_raw.get("mel_fmax_hz", 22050.0),
        pitch_probe_op=tr_raw.get("pitch_probe_op", "pitch_shift"),
        rhythm_probe_op=tr_raw.get("rhythm_probe_op", "rhythm_stretch"),
        timbre_probe_op=tr_raw.get("timbre_probe_op", "timbre_eq_tilt"),
        w_pitch=tr_raw.get("w_pitch", 1.0),
        w_rhythm=tr_raw.get("w_rhythm", 1.0),
        w_timbre=tr_raw.get("w_timbre", 1.0),
        log_every=tr_raw.get("log_every", 50),
        val_every=tr_raw.get("val_every", 1000),
        ckpt_every=tr_raw.get("ckpt_every", 2000),
        out_dir=Path(args.out_dir or tr_raw.get("out_dir", "./runs/stage3")),
        keep_last=tr_raw.get("keep_last", 3),
        wandb_project=tr_raw.get("wandb_project"),
        wandb_run_name=tr_raw.get("wandb_run_name"),
        wandb_entity=tr_raw.get("wandb_entity"),
        wandb_mode=args.wandb_mode or tr_raw.get("wandb_mode", "online"),
        encoder_ckpt=encoder_ckpt,
        device=tr_raw.get("device", "cuda"),
        amp=tr_raw.get("amp", True),
        seed=seed,
    )

    trainer = Stage3Trainer(tcfg, ae, ayi, train_loader, val_loader)
    trainer.fit()


if __name__ == "__main__":
    main()
