## Batch Reconstruction

```bash
cd cloud_code/part_1
python -m evaluation.reconstruct \
  --config configs/train_variant_b.yaml \
  --ckpt runs/stage1_variant_b_codec/ckpt_best.pt \
  --split val \
  --max_items 16
```

The same implementation is also exposed through the legacy wrapper:

```bash
python inference/infer.py --config configs/train_variant_b.yaml --ckpt runs/stage1_variant_b_codec/ckpt_best.pt
```

## Outputs

By default, outputs are written under `cloud_code/part_1/eval_outputs/`:

- `metrics.jsonl`: one JSON record per evaluated item
- `metrics.json`: the same records as an array
- `metrics_summary.json`: mean/std summary
- `audio/*_ref.wav` and `audio/*_recon.wav`: reference/reconstruction pairs
- `plots/*_mel.png`: mel-spectrogram comparison
- `plots/*_waveform.png`: waveform comparison
- `plots/metrics_curves.png`: cheap reconstruction metric curves

Current metrics include legacy sanity checks (`snr`, `mse`, `l1`,
`peak_abs_err`) plus the training-aligned reconstruction metrics
(`recon_total`, `recon_sc`, `recon_log_mag`, `recon_log_mel`).

## Dashboard

```bash
cd cloud_code/part_1
streamlit run evaluation/dashboard.py
```

The old dashboard path is also kept as a wrapper:

```bash
streamlit run inference/app.py
```
