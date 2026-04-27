"""
Shared runtime builders for SHAYI Part-1 training/evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from ae_variant_a import VariantAMultiHead, VariantAConfig
from ae_variant_a_codec import VariantACodec, VariantACodecConfig
from ae_variant_b_codec import VariantBCodec, VariantBCodecConfig
from ae_variant_b_seanet import VariantBSEANet, VariantBSEANetConfig
from dataset import Stage1AEDataset, Stage1DataConfig, stage1_collate
from transforms import build_aux_bundle


def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(model_cfg: dict):
    """Build an AE from the YAML ``model`` block (Variant A or B)."""
    variant = model_cfg.get("variant", "B_CODEC").upper().replace("-", "_")
    if variant in {"B_CODEC", "BCODEC", "CODEC", "B_CODEC_V2", "BCODEC_V2", "B_V2", "V2"}:
        valid_fields = {f.name for f in VariantBCodecConfig.__dataclass_fields__.values()}
        v2_kwargs = {k: v for k, v in model_cfg.items() if k != "variant" and k in valid_fields}
        dropped = set(model_cfg.keys()) - valid_fields - {"variant"}
        if dropped:
            print(f"[build_model] B-Codec: dropping unknown keys {sorted(dropped)}")
        cfg = VariantBCodecConfig(**v2_kwargs)
        return VariantBCodec(cfg)

    if variant in {"A", "A_MULTIHEAD", "MULTIHEAD", "FEATURE"}:
        valid_fields = {f.name for f in VariantAConfig.__dataclass_fields__.values()}
        a_kwargs = {k: v for k, v in model_cfg.items() if k != "variant" and k in valid_fields}
        dropped = set(model_cfg.keys()) - valid_fields - {"variant"}
        if dropped:
            print(f"[build_model] Variant-A: dropping unknown keys {sorted(dropped)}")
        cfg = VariantAConfig(**a_kwargs)
        return VariantAMultiHead(cfg)

    if variant in {"B_SEANET", "BSEANET"}:
        valid_fields = {f.name for f in VariantBSEANetConfig.__dataclass_fields__.values()}
        bs_kwargs = {k: v for k, v in model_cfg.items() if k != "variant" and k in valid_fields}
        dropped = set(model_cfg.keys()) - valid_fields - {"variant"}
        if dropped:
            print(f"[build_model] B-SEANet: dropping unknown keys {sorted(dropped)}")
        cfg = VariantBSEANetConfig(**bs_kwargs)
        return VariantBSEANet(cfg)

    if variant in {"A_CODEC", "ACODEC", "A_DAC", "HYBRID"}:
        valid_fields = {f.name for f in VariantACodecConfig.__dataclass_fields__.values()}
        ac_kwargs = {k: v for k, v in model_cfg.items() if k != "variant" and k in valid_fields}
        dropped = set(model_cfg.keys()) - valid_fields - {"variant"}
        if dropped:
            print(f"[build_model] Variant-A-Codec: dropping unknown keys {sorted(dropped)}")
        cfg = VariantACodecConfig(**ac_kwargs)
        return VariantACodec(cfg)

    raise ValueError(f"Unknown AE variant: {variant}")


def build_dataset(data_cfg: dict, split: str = "val", *, training: bool = False) -> Stage1AEDataset:
    """Build the shared Stage-1-style dataset for train/eval splits."""
    list_path = data_cfg.get(f"{split}_list")
    cfg = Stage1DataConfig(
        roots=[Path(r) for r in data_cfg.get("roots", [])],
        track_list=Path(list_path) if list_path else None,
        sample_rate=data_cfg.get("sample_rate", 44100),
        crop_seconds=data_cfg.get("crop_seconds", 10.0),
        feat_hop=data_cfg.get("feat_hop", 512),
        latent_hop=data_cfg.get("latent_hop", 512),
        normalize_audio=data_cfg.get("normalize_audio", True),
        augment=data_cfg.get("augment", False) and training and split == "train",
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


def build_dataloader(
    data_cfg: dict,
    split: str = "val",
    *,
    batch_size: int = 4,
    num_workers: int = 0,
    training: bool = False,
) -> DataLoader:
    ds = build_dataset(data_cfg, split=split, training=training)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        collate_fn=stage1_collate,
        drop_last=training,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def load_ae_checkpoint(model, ckpt_path: Path | str, *, map_location="cpu", strict: bool = True) -> dict:
    """Load a Stage 1/2 AE checkpoint. Returns the raw checkpoint dict."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"AE checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    model_state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(model_state, strict=strict)
    return state if isinstance(state, dict) else {"model": state}


def forward_ae_batch(model, batch: Dict, device: torch.device) -> Dict[str, Tensor]:
    """Forward a collated batch through Variant A or B with correct routing."""
    audio = batch["audio"].to(device, non_blocking=True)
    input_mode = getattr(model, "input_mode", "waveform")

    if input_mode == "feature":
        enc = model(
            batch["feat_pitch"].to(device, non_blocking=True),
            batch["feat_rhythm"].to(device, non_blocking=True),
            batch["feat_mel"].to(device, non_blocking=True),
        )
    elif bool(getattr(model, "uses_aux_adapters", False)):
        aux_bundle = build_aux_bundle(
            batch,
            device=device,
            is_variant_b=True,
            aux_zero_mask=getattr(model.cfg, "aux_zero_mask", ()),
        )
        enc = model(audio, *aux_bundle.encode_args())
    else:
        enc = model(audio)

    enc["__audio"] = audio
    return enc


def align_audio_pair(y_hat: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Crop prediction/target to their shared time length."""
    T = min(y_hat.shape[-1], y.shape[-1])
    return y_hat[..., :T], y[..., :T]


def default_eval_out_dir(config_path: Path | str, ckpt_path: Optional[Path | str], split: str) -> Path:
    config_stem = Path(config_path).stem
    ckpt_stem = Path(ckpt_path).stem if ckpt_path else "no_ckpt"
    return Path(__file__).resolve().parent / "eval_outputs" / f"{config_stem}_{ckpt_stem}_{split}"
