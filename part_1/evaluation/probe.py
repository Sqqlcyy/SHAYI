"""Canonical disentanglement + reconstruction probe.

Replaces the ad-hoc ``/root/probe_*.py`` scripts that were scattered on
the training box. Single entry point for any SHAYI variant (A / A-Codec /
B-Codec). Outputs go under ``<out_dir>/probe/``.

Metrics computed:
    - Leakage R² (linear probe z → aux), via ``losses.leakage.linear_probe_r2``
    - MINE MI (nats), via ``losses.leakage.mine_mi_nats``
    - Per-sample recon: SNR / SI-SDR / log-STFT L1 / mel-dB L1
      via ``evaluation.metrics.per_sample_recon_metrics``
    - Distribution FAD (VGGish / PANN / CLAP) via ``evaluation.metrics.FADComputer``

CLI:
    python -m evaluation.probe --cfg configs/train_a_codec.yaml [--n 32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Pin threads before torch import to dodge librosa/numba deadlocks.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import soundfile as sf
import torch
import yaml

_PART1_ROOT = Path(__file__).resolve().parents[1]
if str(_PART1_ROOT) not in sys.path:
    sys.path.insert(0, str(_PART1_ROOT))

from dataset import Stage1AEDataset, Stage1DataConfig  # noqa: E402
from evaluation.metrics import FADComputer, per_sample_recon_metrics  # noqa: E402
from losses.leakage import linear_probe_r2, mine_mi_nats  # noqa: E402


# ---------------------------------------------------------------------------
# Variant dispatch (matches runtime.build_model's keyword set)
# ---------------------------------------------------------------------------
_A_KEYS = {"A", "A_MULTIHEAD", "MULTIHEAD", "FEATURE"}
_A_CODEC_KEYS = {"A_CODEC", "ACODEC", "A_DAC", "HYBRID"}
_B_KEYS = {"B_CODEC", "BCODEC", "CODEC", "B_CODEC_V2", "B_V2", "V2"}


def _build_model(raw: Dict) -> Tuple[torch.nn.Module, str]:
    variant = raw["model"].get("variant", "B_CODEC").upper().replace("-", "_")
    mc = raw["model"]
    if variant in _A_KEYS:
        from ae_variant_a import VariantAMultiHead, VariantAConfig
        valid = {f.name for f in VariantAConfig.__dataclass_fields__.values()}
        kw = {k: v for k, v in mc.items() if k != "variant" and k in valid}
        return VariantAMultiHead(VariantAConfig(**kw)), variant
    if variant in _A_CODEC_KEYS:
        from ae_variant_a_codec import VariantACodec, VariantACodecConfig
        valid = {f.name for f in VariantACodecConfig.__dataclass_fields__.values()}
        kw = {k: v for k, v in mc.items() if k != "variant" and k in valid}
        return VariantACodec(VariantACodecConfig(**kw)), variant
    if variant in _B_KEYS:
        from ae_variant_b_codec import VariantBCodec, VariantBCodecConfig
        valid = {f.name for f in VariantBCodecConfig.__dataclass_fields__.values()}
        kw = {k: v for k, v in mc.items() if k != "variant" and k in valid}
        return VariantBCodec(VariantBCodecConfig(**kw)), variant
    raise ValueError(f"unknown variant {variant}")


def _build_dataset(raw: Dict) -> Stage1AEDataset:
    d = raw["data"]
    val_list = d.get("val_list")
    if val_list is None:
        raise SystemExit("probe requires val_list in data config")
    return Stage1AEDataset(Stage1DataConfig(
        roots=[Path(r) for r in d.get("roots", [])],
        track_list=Path(val_list) if not Path(val_list).is_absolute()
        else Path(val_list),
        sample_rate=d.get("sample_rate", 44100),
        crop_seconds=d.get("crop_seconds", 10.0),
        feat_hop=d.get("feat_hop", 512),
        latent_hop=d.get("latent_hop", 512),
        normalize_audio=True, augment=False,
        pitch_feat_name=d.get("pitch_feat_name", "pitch_salience_instru_nondrum.npy"),
        rhythm_multi_feat_name=d.get("rhythm_multi_feat_name", "rhythm_multi_instru.npy"),
        spec_feat_name=d.get("spec_feat_name", "spec_instru.npy"),
        mel_linear_feat_name=d.get("mel_linear_feat_name", "mel_linear_instru.npy"),
        envelope_feat_name=d.get("envelope_feat_name", "envelope_instru.npy"),
        prefer_envelope_timbre=d.get("prefer_envelope_timbre", False),
        timbre_feat_name=d.get("timbre_feat_name", "mfcc_instru.npy"),
        rhythm_feat_name=d.get("rhythm_feat_name", "rhythm_instru.npy"),
        n_pitch_bins=d.get("n_pitch_bins", 588),
        n_mels=d.get("n_mels", 128),
        n_mfcc=d.get("n_mfcc", 20),
        n_rhythm_channels=d.get("n_rhythm_channels", 8),
        n_envelope_dim=d.get("n_envelope_dim", 80),
        allow_missing_aux=False,
    ))


@torch.no_grad()
def _forward_once(model, sample, variant: str):
    fp = sample["feat_pitch"].unsqueeze(0).cuda()
    fr = sample["feat_rhythm"].unsqueeze(0).cuda()
    fm = sample["feat_mel"].unsqueeze(0).cuda()
    audio = sample["audio"].unsqueeze(0).cuda()
    aux_p = sample["aux_pitch"].unsqueeze(0).cuda()
    aux_r = sample["aux_rhythm"].unsqueeze(0).cuda()
    aux_t = sample["aux_timbre"].unsqueeze(0).cuda()

    if variant in _A_KEYS:
        env = aux_t if getattr(model, "uses_envelope_aux", False) else None
        kw = {}
        if getattr(model, "uses_adv_kwargs", False):
            kw = dict(aux_pitch=aux_p, aux_rhythm=aux_r, aux_timbre=aux_t)
        return model(fp, fr, fm, env, **kw), audio
    if variant in _A_CODEC_KEYS:
        env = aux_t if model.uses_envelope_aux else None
        return model(fp, fr, fm, env,
                     aux_pitch=aux_p, aux_rhythm=aux_r, aux_timbre=aux_t,
                     audio=audio), audio
    # B_CODEC
    return model(audio, aux_p, aux_r, aux_t), audio


def _materialize_lazy_params(model, variant: str) -> None:
    """Force any lazy-init submodules (Snake alpha/beta) before load_state_dict."""
    cfg = model.cfg
    with torch.no_grad():
        try:
            dp = torch.zeros(1, getattr(cfg, "pitch_in_dim", 588), 10).cuda()
            dr = torch.zeros(1, getattr(cfg, "rhythm_in_dim", 8), 10).cuda()
            dm = torch.zeros(1, getattr(cfg, "mel_in_dim", 128), 10).cuda()
            env = (
                torch.zeros(1, getattr(cfg, "envelope_in_dim", 80), 10).cuda()
                if getattr(cfg, "use_envelope_aux", False) else None
            )
            if variant in _A_KEYS:
                model(dp, dr, dm, env)
            elif variant in _A_CODEC_KEYS:
                model(dp, dr, dm, env, audio=torch.zeros(1, 1, 44100).cuda())
        except Exception as e:  # noqa: BLE001
            print(f"[probe] dummy forward skipped ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Defaults to latest ckpt_step*.pt under train.out_dir.")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Defaults to <train.out_dir>/probe/.")
    ap.add_argument("--n", type=int, default=200, help="Samples for stats (FAD needs 200+ for stability).")
    ap.add_argument("--n_audio", type=int, default=6, help="Audio pairs to dump.")
    ap.add_argument("--fad_backbone", type=str, default="vggish")
    ap.add_argument("--skip_fad", action="store_true")
    ap.add_argument("--mi_inner_steps", type=int, default=300)
    args = ap.parse_args()

    raw = yaml.safe_load(args.cfg.read_text())
    run_dir = Path(raw["train"]["out_dir"])
    ckpt = args.ckpt
    if ckpt is None:
        ckpts = sorted([c for c in run_dir.glob("ckpt_step*.pt") if "_ema" not in c.stem])
        if not ckpts:
            raise SystemExit(f"no ckpts in {run_dir}")
        ckpt = ckpts[-1]
    out_dir = args.out_dir or (run_dir / "probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] cfg={args.cfg.name}  ckpt={ckpt.name}  out={out_dir}")

    model, variant = _build_model(raw)
    model = model.cuda()
    _materialize_lazy_params(model, variant)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    ds = _build_dataset(raw)
    n = min(args.n, len(ds))
    print(f"[probe] gathering {n} samples (B=1)...")

    buckets = {k: [] for k in ("z_p", "z_r", "z_t", "z_content",
                               "aux_pitch", "aux_rhythm", "aux_timbre")}
    recon_records = []
    sr = raw["data"].get("sample_rate", 44100)
    ref_dir = out_dir / "fad_ref"
    rec_dir = out_dir / "fad_rec"
    ref_dir.mkdir(exist_ok=True)
    rec_dir.mkdir(exist_ok=True)

    for i in range(n):
        sample = ds[i]
        out, audio = _forward_once(model, sample, variant)
        buckets["z_p"].append(out["z_p"].cpu())
        buckets["z_r"].append(out["z_r"].cpu())
        buckets["z_t"].append(out["z_t"].cpu())
        buckets["z_content"].append(out.get("z_content", out["z_r"]).cpu())
        buckets["aux_pitch"].append(sample["aux_pitch"].unsqueeze(0).cpu())
        buckets["aux_rhythm"].append(sample["aux_rhythm"].unsqueeze(0).cpu())
        buckets["aux_timbre"].append(sample["aux_timbre"].unsqueeze(0).cpu())

        if "x_hat" in out:
            x = audio.cpu().numpy()[0, 0]
            y = out["x_hat"].cpu().numpy()[0, 0]
            m = per_sample_recon_metrics(x, y, sample_rate=sr)
            m["track"] = Path(ds.sample_dirs[i]).name.replace(" ", "_")[:40]
            recon_records.append(m)
            sf.write(ref_dir / f"{i:03d}.wav", x[:min(len(x), len(y))], sr)
            sf.write(rec_dir / f"{i:03d}.wav", y[:min(len(x), len(y))], sr)

    stacked = {k: torch.cat(v, dim=0) for k, v in buckets.items()}

    # --- Leakage R² ---
    print("\n=== Leakage R² ===")
    r2: Dict[str, float] = {}
    for zk in ("z_p", "z_r", "z_t", "z_content"):
        for ak, nm in (("aux_pitch", "pitch"), ("aux_rhythm", "rhythm"), ("aux_timbre", "envelope")):
            v = linear_probe_r2(stacked[zk], stacked[ak])
            r2[f"{zk}_to_{nm}"] = float(v.item()) if v == v else float("nan")
    for k in sorted(r2):
        print(f"  {k:30s} R²= {r2[k]:+.4f}")

    # --- MINE MI ---
    print("\n=== MINE MI (nats) ===")
    mi: Dict[str, float] = {}
    for ak, bk in (
        ("z_p", "z_r"), ("z_p", "z_t"), ("z_r", "z_t"),
        ("z_content", "z_p"), ("z_content", "z_r"), ("z_content", "z_t"),
    ):
        v = mine_mi_nats(stacked[ak].cuda(), stacked[bk].cuda(),
                         inner_steps=args.mi_inner_steps)
        mi[f"{ak}__{bk}"] = v
        print(f"  {ak:10s} ↔ {bk:10s} MI= {v:+.4f}")

    # --- Reconstruction ---
    recon_agg: Dict[str, float] = {}
    if recon_records:
        print("\n=== Per-sample recon (mean over samples) ===")
        keys = ("snr_db", "si_sdr", "log_stft_l1", "mel_db_l1")
        recon_agg = {k: float(np.mean([r[k] for r in recon_records])) for k in keys}
        for k in keys:
            print(f"  mean_{k:15s}= {recon_agg[k]:+.4f}")

    # --- FAD ---
    fad_scores: Dict[str, float] = {}
    if not args.skip_fad and recon_records:
        print(f"\n=== FAD ({args.fad_backbone}) ===")
        try:
            fad = FADComputer(model_name=args.fad_backbone,
                              sample_rate=16000 if args.fad_backbone == "vggish" else sr)
            fad_scores = fad.compute(ref_dir, rec_dir)
            for k, v in fad_scores.items():
                print(f"  {k}= {v:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAD failed: {type(e).__name__}: {e}")

    # Dump first n_audio pairs at the out_dir root for easy listening
    for i, rec in enumerate(recon_records[: args.n_audio]):
        src = rec["track"]
        ref_src = ref_dir / f"{i:03d}.wav"
        rec_src = rec_dir / f"{i:03d}.wav"
        if ref_src.exists():
            (out_dir / f"{i:02d}_{src}_orig.wav").write_bytes(ref_src.read_bytes())
            (out_dir / f"{i:02d}_{src}_recon.wav").write_bytes(rec_src.read_bytes())

    summary = {
        "ckpt": str(ckpt),
        "variant": variant,
        "n_samples": n,
        "leakage_R2": r2,
        "mi_nats": mi,
        "recon_agg": recon_agg,
        "recon_per_sample": recon_records,
        "fad": fad_scores,
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[probe] summary -> {out_dir}/probe_summary.json")


if __name__ == "__main__":
    main()
