#!/bin/bash
# SHAYI part_2 directory bootstrap
# Usage:
#   bash setup_shayi_dir.sh
#
# This creates the full folder layout for SHAYI DiT + synthesizers + training + inference.

set -e

ROOT="src"

echo "[SHAYI] Creating directory tree under ./${ROOT}"

# ---- top level ----
mkdir -p "${ROOT}"
touch "${ROOT}/__init__.py"

# ---- data ----
mkdir -p "${ROOT}/data"
touch "${ROOT}/data/__init__.py"
touch "${ROOT}/data/latent_dataset.py"
touch "${ROOT}/data/stem_mel_dataset.py"

# ---- models ----
mkdir -p "${ROOT}/models"
touch "${ROOT}/models/__init__.py"
touch "${ROOT}/models/shayi_dit.py"
touch "${ROOT}/models/synth_perc.py"
touch "${ROOT}/models/synth_nonperc.py"
touch "${ROOT}/models/mix_module.py"

# ---- losses ----
mkdir -p "${ROOT}/losses"
touch "${ROOT}/losses/__init__.py"
touch "${ROOT}/losses/flow_matching.py"
touch "${ROOT}/losses/mel_recon.py"

# ---- training ----
mkdir -p "${ROOT}/training"
touch "${ROOT}/training/__init__.py"
touch "${ROOT}/training/train_dit.py"
touch "${ROOT}/training/train_synth_perc.py"
touch "${ROOT}/training/train_synth_nonperc.py"
touch "${ROOT}/training/train_mix.py"

# ---- inference ----
mkdir -p "${ROOT}/inference"
touch "${ROOT}/inference/__init__.py"
touch "${ROOT}/inference/sample_dit.py"
touch "${ROOT}/inference/full_pipeline.py"
touch "${ROOT}/inference/edit2x.py"

# ---- checkpoints / runs / logs ----
mkdir -p runs
mkdir -p runs/shayi_dit_baseline
mkdir -p runs/shayi_dit_hrca
mkdir -p runs/synth_perc
mkdir -p runs/synth_nonperc
mkdir -p runs/mix_module
mkdir -p runs/demos

# ---- configs ----
mkdir -p configs
touch configs/dit_baseline.yaml
touch configs/dit_hrca.yaml
touch configs/synth_perc.yaml
touch configs/synth_nonperc.yaml
touch configs/mix.yaml

# ---- scripts ----
mkdir -p scripts
touch scripts/dump_stem_mel.py
touch scripts/eval_edit2x.py
touch scripts/eval_designness.py
touch scripts/sanity_check.py

# ---- docs ----
mkdir -p docs
touch docs/ARCH.md
touch docs/CONTRIBUTIONS.md
touch docs/DATA_SPEC.md

echo "[SHAYI] Done."
echo ""
echo "Resulting structure:"
find "${ROOT}" configs scripts runs docs -type d -o -type f | sort
