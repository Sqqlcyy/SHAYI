from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.metrics import ReconstructionMetricComputer, summarize_records, waveform_metrics_np  # noqa: E402
from evaluation.visualize import plot_mel_pair, plot_metric_curves, plot_waveform_pair, save_wav  # noqa: E402
from runtime import (  # noqa: E402
    align_audio_pair,
    build_dataloader,
    build_model,
    default_eval_out_dir,
    forward_ae_batch,
    load_ae_checkpoint,
    load_yaml,
)


def _slug(text: str, fallback: str) -> str:
    text = str(text or fallback)
    parts = Path(text).parts[-3:]
    s = "_".join(parts) if parts else text
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s[:120] or fallback


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_train_metrics(raw: dict) -> dict:
    train = raw.get("train", {})
    return {
        "fft_sizes": tuple(train.get("fft_sizes", (1024, 2048, 4096))),
        "hop_sizes": tuple(train.get("hop_sizes", (120, 240, 480))),
        "win_lengths": tuple(train.get("win_lengths", (960, 1920, 3840))),
        "n_mels": int(train.get("n_mels", 128)),
        "w_logmel": float(train.get("w_logmel", 0.5)),
    }


def run_reconstruction(args: argparse.Namespace) -> Path:
    raw = load_yaml(args.config)
    data_cfg = raw["data"]
    model_cfg = raw.get("model") or raw.get("ae")
    if model_cfg is None:
        raise KeyError("Config must contain a `model:` or `ae:` block.")

    out_dir = Path(args.out_dir) if args.out_dir else default_eval_out_dir(args.config, args.ckpt, args.split)
    audio_dir = out_dir / "audio"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(model_cfg).to(device)
    state = load_ae_checkpoint(model, args.ckpt, map_location=device, strict=not args.non_strict)
    model.eval()

    train_cfg = raw.get("train", {})
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 4))
    num_workers = int(args.num_workers if args.num_workers is not None else 0)
    loader = build_dataloader(data_cfg, args.split, batch_size=batch_size, num_workers=num_workers, training=False)

    metric_cfg = _load_train_metrics(raw)
    sample_rate = int(data_cfg.get("sample_rate", 32000))
    metric_computer = ReconstructionMetricComputer(sample_rate=sample_rate, device=device, **metric_cfg)

    records: List[Dict] = []
    processed = 0
    max_items = int(args.max_items or 0)
    ckpt_step = state.get("step", None)
    if torch.is_tensor(ckpt_step):
        ckpt_step = int(ckpt_step.detach().cpu().item())

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc="reconstruct")):
            enc = forward_ae_batch(model, batch, device)
            y_hat, y = align_audio_pair(enc["x_hat"], enc["__audio"])
            B = int(y.shape[0])

            for i in range(B):
                if max_items and processed >= max_items:
                    break
                ref_t = y[i : i + 1].detach()
                rec_t = y_hat[i : i + 1].detach()
                ref_np = ref_t.squeeze().cpu().numpy().astype(np.float32)
                rec_np = rec_t.squeeze().cpu().numpy().astype(np.float32)

                item_name = _slug(batch.get("track_id", [""] * B)[i], f"item_{processed:05d}")
                prefix = f"{processed:05d}_{item_name}"
                ref_rel = Path("audio") / f"{prefix}_ref.wav"
                rec_rel = Path("audio") / f"{prefix}_recon.wav"
                mel_rel = Path("plots") / f"{prefix}_mel.png"
                wav_rel = Path("plots") / f"{prefix}_waveform.png"

                if args.save_audio:
                    save_wav(out_dir / ref_rel, ref_np, sample_rate)
                    save_wav(out_dir / rec_rel, rec_np, sample_rate)

                if args.save_plots and processed < args.plot_items:
                    plot_mel_pair(ref_np, rec_np, sample_rate=sample_rate, out_path=out_dir / mel_rel, title=prefix)
                    plot_waveform_pair(ref_np, rec_np, sample_rate=sample_rate, out_path=out_dir / wav_rel, title=prefix)

                metrics = waveform_metrics_np(ref_np, rec_np)
                metrics.update(metric_computer.compute(rec_t, ref_t))
                record = {
                    "index": processed,
                    "batch_index": batch_index,
                    "track_id": batch.get("track_id", [""] * B)[i],
                    "model_type": type(model).__name__,
                    "input_mode": getattr(model, "input_mode", "waveform"),
                    "ckpt_step": ckpt_step,
                    **metrics,
                    "ref_wav": str(ref_rel) if args.save_audio else "",
                    "recon_wav": str(rec_rel) if args.save_audio else "",
                    "mel_plot": str(mel_rel) if args.save_plots and processed < args.plot_items else "",
                    "waveform_plot": str(wav_rel) if args.save_plots and processed < args.plot_items else "",
                }
                records.append(record)
                processed += 1

            if max_items and processed >= max_items:
                break

    summary = summarize_records(records)
    summary.update(
        {
            "config": str(args.config),
            "ckpt": str(args.ckpt),
            "split": args.split,
            "sample_rate": sample_rate,
            "model_type": type(model).__name__,
            "input_mode": getattr(model, "input_mode", "waveform"),
            "ckpt_step": ckpt_step,
        }
    )

    _write_jsonl(out_dir / "metrics.jsonl", records)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_metric_curves(records, plot_dir / "metrics_curves.png")

    print(f"[eval] wrote {len(records)} records to {out_dir}")
    print(f"[eval] summary: {out_dir / 'metrics_summary.json'}")
    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch AE reconstruction/evaluation.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--max_items", type=int, default=0, help="0 means evaluate the whole split.")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--non_strict", action="store_true", help="Load checkpoint with strict=False.")
    ap.add_argument("--save_audio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--plot_items", type=int, default=8)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    run_reconstruction(args)


if __name__ == "__main__":
    main()
