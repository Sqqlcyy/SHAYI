"""Streamlit dashboard for outputs produced by ``evaluation.reconstruct``."""
from __future__ import annotations

import json
from pathlib import Path


def _load_records(out_dir: Path):
    metrics_json = out_dir / "metrics.json"
    if metrics_json.exists():
        with open(metrics_json, "r", encoding="utf-8") as f:
            return json.load(f)
    metrics_jsonl = out_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        with open(metrics_jsonl, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return []


def _load_summary(out_dir: Path):
    p = out_dir / "metrics_summary.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(out_dir: Path, rel: str) -> Path:
    return out_dir / rel if rel else Path()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SHAYI AE Evaluation", layout="wide")
    st.title("SHAYI AE Evaluation Dashboard")

    default_dir = str(Path(__file__).resolve().parents[1] / "eval_outputs")
    out_dir_text = st.sidebar.text_input("Evaluation output directory", default_dir)
    out_dir = Path(out_dir_text).expanduser()
    if out_dir.name == "eval_outputs" and not (out_dir / "metrics.json").exists():
        candidates = sorted([p for p in out_dir.glob("*") if (p / "metrics.json").exists()])
        if candidates:
            out_dir = candidates[-1]
            st.sidebar.info(f"Using latest run: {out_dir}")

    records = _load_records(out_dir)
    summary = _load_summary(out_dir)
    if not records:
        st.warning(
            "No metrics found. Run `python -m evaluation.reconstruct --config ... --ckpt ...` first."
        )
        return

    st.caption(f"Output: `{out_dir}`")
    cols = st.columns(4)
    cols[0].metric("Items", str(summary.get("num_items", len(records))))
    cols[1].metric("SNR mean", f"{summary.get('snr/mean', float('nan')):.2f} dB")
    cols[2].metric("MSE mean", f"{summary.get('mse/mean', float('nan')):.4g}")
    cols[3].metric("Recon total", f"{summary.get('recon_total/mean', float('nan')):.3f}")

    curve = out_dir / "plots" / "metrics_curves.png"
    if curve.exists():
        st.subheader("Metric Curves")
        st.image(str(curve))

    st.subheader("Item Browser")
    labels = [f"{r['index']:05d} | {Path(str(r.get('track_id', ''))).name}" for r in records]
    selected = st.selectbox("Select item", options=list(range(len(records))), format_func=lambda i: labels[i])
    rec = records[selected]

    mcols = st.columns(5)
    for col, key in zip(mcols, ["snr", "mse", "l1", "recon_total", "recon_log_mel"]):
        val = rec.get(key, float("nan"))
        col.metric(key, f"{float(val):.4g}")

    audio_cols = st.columns(2)
    ref_wav = _resolve(out_dir, rec.get("ref_wav", ""))
    recon_wav = _resolve(out_dir, rec.get("recon_wav", ""))
    with audio_cols[0]:
        st.markdown("**Reference**")
        if ref_wav.exists():
            st.audio(str(ref_wav), format="audio/wav")
    with audio_cols[1]:
        st.markdown("**Reconstruction**")
        if recon_wav.exists():
            st.audio(str(recon_wav), format="audio/wav")

    plot_cols = st.columns(2)
    mel_plot = _resolve(out_dir, rec.get("mel_plot", ""))
    waveform_plot = _resolve(out_dir, rec.get("waveform_plot", ""))
    with plot_cols[0]:
        if mel_plot.exists():
            st.image(str(mel_plot), caption="Mel comparison")
    with plot_cols[1]:
        if waveform_plot.exists():
            st.image(str(waveform_plot), caption="Waveform comparison")

    with st.expander("Raw record"):
        st.json(rec)


if __name__ == "__main__":
    main()
