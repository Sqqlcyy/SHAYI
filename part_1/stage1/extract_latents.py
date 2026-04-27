import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import Stage1AEDataset, Stage1DataConfig, stage1_collate
from runtime import build_model

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
        augment=False,
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
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(raw["model"])
    
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
        
    model.to(device)
    model.eval()

    is_variant_b = bool(getattr(model, "uses_aux_adapters", False))
    input_mode = getattr(model, "input_mode", "waveform")
    needs_target_wav = bool(getattr(model, "needs_target_wav", False))

    if is_variant_b:
        from transforms import build_aux_bundle

    dataset = _build_dataset(raw["data"], args.split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=stage1_collate,
        drop_last=False,
        pin_memory=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
        for batch in tqdm(loader):
            audio = batch["audio"].to(device, non_blocking=True)
            
            if input_mode == "feature":
                feat_pitch = batch["feat_pitch"].to(device, non_blocking=True)
                feat_rhythm = batch["feat_rhythm"].to(device, non_blocking=True)
                feat_mel = batch["feat_mel"].to(device, non_blocking=True)
                wants_env = bool(getattr(model, "uses_envelope_aux", False))
                feat_env = batch["aux_timbre"].to(device, non_blocking=True) if wants_env else None
                extra_args = (feat_env,) if wants_env else ()
                
                is_a_codec = type(model).__name__ == "VariantACodec"
                wants_adv_kwargs = is_a_codec or bool(getattr(model, "uses_adv_kwargs", False))
                extra_kwargs = {}
                if wants_adv_kwargs:
                    extra_kwargs["aux_pitch"] = batch["aux_pitch"].to(device, non_blocking=True)
                    extra_kwargs["aux_rhythm"] = batch["aux_rhythm"].to(device, non_blocking=True)
                    extra_kwargs["aux_timbre"] = batch["aux_timbre"].to(device, non_blocking=True)
                if is_a_codec:
                    extra_kwargs["audio"] = audio
                    
                if needs_target_wav:
                    enc = model(feat_pitch, feat_rhythm, feat_mel, *extra_args, audio, deterministic=True, **extra_kwargs)
                else:
                    enc = model(feat_pitch, feat_rhythm, feat_mel, *extra_args, deterministic=True, **extra_kwargs)
            
            elif is_variant_b:
                aux_bundle = build_aux_bundle(
                    batch,
                    device=device,
                    is_variant_b=True,
                    aux_zero_mask=getattr(model.cfg, "aux_zero_mask", ()),
                )
                enc = model(audio, *list(aux_bundle.encode_args()), deterministic=True)
            else:
                enc = model(audio)

            z_p = enc.get("z_p")
            z_r = enc.get("z_r")
            z_t = enc.get("z_t")

            if z_p is not None and z_p.shape[-1] > args.max_len:
                z_p = z_p[..., :args.max_len]
            if z_r is not None and z_r.shape[-1] > args.max_len:
                z_r = z_r[..., :args.max_len]
            if z_t is not None and z_t.ndim > 2 and z_t.shape[-1] > args.max_len:
                z_t = z_t[..., :args.max_len]

            out_dict = {}
            if z_p is not None:
                out_dict["z_p"] = z_p.cpu().squeeze(0).clone()
            if z_r is not None:
                out_dict["z_r"] = z_r.cpu().squeeze(0).clone()
            if z_t is not None:
                out_dict["z_t"] = z_t.cpu().squeeze(0).clone()

            torch.save(out_dict, args.out_dir / f"latent_{idx:06d}.pt")
            idx += 1

if __name__ == "__main__":
    main()