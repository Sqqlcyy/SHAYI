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
from stage1.trainer import Stage1Trainer, TrainerConfig  # noqa: E402
from stage1.adv_trainer import Stage1AdvTrainer, AdvTrainerConfig  # noqa: E402
# Variant dispatch now lives in runtime.build_model — see _build_model() below.


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(model_cfg: dict):
    from runtime import build_model
    return build_model(model_cfg)


_VARIANT_B_ONLY_TRAIN_FIELDS = (
    "w_codec_distill",
    "w_trunk_l1",
    "aux_dropout_p",
    "w_adv_pitch_on_t",
)
_VARIANT_A_ONLY_MODEL_FIELDS = (
    "dec_ratios", "dec_activation", "dec_activation_params",
    "dec_n_filters", "dec_n_residual_layers", "dec_lstm", "dec_norm",
    "dec_final_activation", "dec_in_dim", "content_residual",
    "enc_hidden", "enc_n_layers_side", "enc_n_layers_main",
    "pitch_in_dim", "rhythm_in_dim", "mel_in_dim",
    "fusion_hidden_dims", "film_gamma_init",
)
_VARIANT_B_ONLY_MODEL_FIELDS = (
    "codec_backend", "codec_model", "codec_bitrate", "codec_dim",
    "codec_sample_rate", "freeze_codec",
    "aux_pitch_dim", "aux_rhythm_dim", "aux_timbre_dim",
    "trunk_out_p", "trunk_out_r", "trunk_out_t",
    "trunk_hidden", "trunk_n_layers", "trunk_kernel",
    "film_fusion_hidden_dims", "film_zt_mode",
    "use_pitch_adversary", "pitch_adversary_lambd", "pitch_adversary_hidden",
    "z_t_noise_std",
)


def _validate_cfg_for_variant(raw: dict) -> None:
    """Guard against silent cross-variant misconfiguration.

    Errors on things that would silently wreck the run (the KL bug on A),
    warns on things that are merely wasteful (orphan knobs). Runs before
    trainer construction so failures abort cleanly.
    """
    variant = (raw.get("model", {}).get("variant", "B_CODEC")).upper().replace("-", "_")
    is_a = variant in {"A", "A_MULTIHEAD", "MULTIHEAD", "FEATURE"}
    is_a_codec = variant in {"A_CODEC", "ACODEC", "A_DAC", "HYBRID"}
    is_b_seanet = variant in {"B_SEANET", "BSEANET"}
    tr = raw.get("train", {}) or {}
    model = raw.get("model", {}) or {}

    if is_a or is_a_codec:
        # Both A variants are deterministic; w_kl > 0 silently drives z_p to 0.
        if float(tr.get("w_kl", 0.0)) > 0.0:
            raise ValueError(
                "Variant A (and A_CODEC) are deterministic (log_var=0); w_kl > 0 "
                "will drive z_p to zero via 0.5·||mu||². Set w_kl: 0.0 in the yaml."
            )

    if is_a:
        # Soft warnings: orphan B-only training knobs.
        ignored = [k for k in _VARIANT_B_ONLY_TRAIN_FIELDS if float(tr.get(k, 0.0) or 0.0) > 0.0]
        if ignored:
            print(f"[validate] Variant A ignores these B-only train fields (non-zero): {ignored}")
        if bool(tr.get("require_codec_distill", False)):
            raise ValueError("Variant A has no codec — set require_codec_distill: false.")
        # Orphan B-only model fields.
        b_model_set = [k for k in _VARIANT_B_ONLY_MODEL_FIELDS if k in model]
        if b_model_set:
            print(f"[validate] Variant A ignores these B-only model fields: {b_model_set}")
    elif is_a_codec:
        # Variant A-Codec needs codec bridge AND doesn't have a trunk.
        if not bool(tr.get("require_codec_distill", True)):
            print("[validate] Variant A-Codec: require_codec_distill=false — codec_distill is the main loss; usually want this true.")
    elif is_b_seanet:
        # B-SEANet should NOT use codec_distill (SEANet defines its own latent space).
        if float(tr.get("w_codec_distill", 0.0) or 0.0) > 0.0:
            print("[validate] B-SEANet: w_codec_distill > 0 is unnecessary — SEANet decoder doesn't need DAC latent distribution. Consider setting to 0.")
    else:
        # Variant B: warn about stray A-only fields in model block.
        a_model_set = [k for k in _VARIANT_A_ONLY_MODEL_FIELDS if k in model]
        if a_model_set:
            print(f"[validate] Variant B ignores these A-only model fields: {a_model_set}")


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
    ap.add_argument("--overfit", type=int, default=0,
                    help="If >0, use only the first N training samples (smoke test).")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Override trainer.max_steps from CLI.")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Override trainer.out_dir from CLI.")
    ap.add_argument("--wandb_mode", type=str, default=None,
                    help="Override wandb_mode (online/offline/disabled).")
    args = ap.parse_args()

    raw = _load_yaml(args.config)
    _validate_cfg_for_variant(raw)

    # Seed
    seed = raw.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data
    train_ds = _build_dataset(raw["data"], "train")
    if args.overfit > 0:
        train_ds.sample_dirs = train_ds.sample_dirs[: args.overfit]
        print(f"[overfit] restricting to {len(train_ds.sample_dirs)} samples")

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

    # Model
    model = _build_model(raw["model"])

    # Trainer config
    tr_raw = raw["train"]
    use_adv = bool(tr_raw.get("use_adv", False))
    trainer_config_cls = AdvTrainerConfig if use_adv else TrainerConfig
    tcfg = trainer_config_cls(
        lr=tr_raw.get("lr", 3e-4),
        betas=tuple(tr_raw.get("betas", (0.9, 0.99))),
        weight_decay=tr_raw.get("weight_decay", 1e-4),
        grad_clip=tr_raw.get("grad_clip", 1.0),
        accum_steps=tr_raw.get("accum_steps", 1),
        warmup_steps=tr_raw.get("warmup_steps", 1000),
        max_steps=args.max_steps or tr_raw.get("max_steps", 100_000),
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
        w_moe_t=tr_raw.get("w_moe_t", tr_raw.get("w_moe", 0.0)),
        w_moe_r=tr_raw.get("w_moe_r", 0.0),
        moe_k=tr_raw.get("moe_k", 8),
        w_codec_distill=tr_raw.get("w_codec_distill", 1.0),
        w_trunk_l1=tr_raw.get("w_trunk_l1", 0.0),
        trunk_l1_form=tr_raw.get("trunk_l1_form", "group_lasso"),
        aux_dropout_p=tr_raw.get("aux_dropout_p", 0.0),
        w_pitch_anchor=tr_raw.get("w_pitch_anchor", 0.0),
        pitch_anchor_warmup=tr_raw.get("pitch_anchor_warmup", 1500),
        pitch_anchor_decay_end=tr_raw.get("pitch_anchor_decay_end", 5000),
        pitch_anchor_tau=tr_raw.get("pitch_anchor_tau", 0.5),
        sample_rate=raw["data"].get("sample_rate", 32000),
        fft_sizes=tuple(tr_raw.get("fft_sizes", (1024, 2048, 4096))),
        hop_sizes=tuple(tr_raw.get("hop_sizes", (120, 240, 480))),
        win_lengths=tuple(tr_raw.get("win_lengths", (960, 1920, 3840))),
        n_mels=tr_raw.get("n_mels", 128),
        w_logmel=tr_raw.get("w_logmel", 0.5),
        w_wav_l1=tr_raw.get("w_wav_l1", 1.0),
        log_every=tr_raw.get("log_every", 50),
        val_every=tr_raw.get("val_every", 1000),
        ckpt_every=tr_raw.get("ckpt_every", 2000),
        media_every=tr_raw.get("media_every", 1000),
        out_dir=Path(args.out_dir or tr_raw.get("out_dir", "./runs/stage1")),
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
    )

    if use_adv:
        # Attach adv-only fields onto the AdvTrainerConfig instance.
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

    # One-shot sanity print: latent shapes
    with torch.no_grad():
        sample = next(iter(train_loader))
        enc = trainer._forward_model(sample)
        print("[sanity] latent shapes:")
        for k in ("z_p", "z_r", "z_t", "x_hat"):
            print(f"  {k}: {tuple(enc[k].shape)}")

    trainer.fit()


if __name__ == "__main__":
    main()
