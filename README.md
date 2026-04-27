# 🎛️ SHAYI: The Producer-Aligned Post-Production Engine for AI Music

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](#) 
[![Demo](https://img.shields.io/badge/Demo-HuggingFace-yellow.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#)

**SHAYI (Self-Supervised Hierarchical Analytic Yielding Instructor)** is a paradigm-shifting controllable music editing framework.

Recent foundational Audio-to-Audio and Text-to-Audio models (e.g., Suno, Udio, AudioLDM) have achieved breathtaking zero-to-one generation capabilities. However, professional music production requires **one-to-N secondary development**: the ability to surgically tweak a drum groove, shift a harmony, or brighten a timbre without regenerating the entire track.

**SHAYI is designed to eat all AIMG outputs and make them editable.** 
Whether it's a generated track from Suno or a real studio recording, SHAYI ingests the audio, disentangles its physical factors, and grants you DAW-level parametric control. We build the "Figma layer" on top of the generative AI music ecosystem.

## 🌟 Core Innovations

1. **Analytic Equivariant Autoencoder ($AE^2$)**: Supervised by the Analytic Yielding Instructor (AYI), our encoder enforces strict equivariance, ironing out the latent space into mathematically orthogonal streams: `Pitch`, `Rhythm`, and `Timbre`.
2. **HiFRA-T (Hierarchical Factor-Routed Attention)**: Our flow-matching DiT backbone employs a Directed Acyclic Graph (DAG) mask ($Timbre \to Rhythm \to Pitch$), respecting the physical causality of music and preventing feature collapse.
3. **FactorLock & Energy Guidance**: A zero-training, prompt-free inference mechanism. Want to double the rhythm density of a Suno-generated track? Lock its Pitch and Timbre, and apply a `2.0x` scalar energy guidance to the Rhythm ODE velocity field. 

---

## 🏗️ Architecture

Instead of a black-box monolithic diffusion process, SHAYI operates on a strict **Analysis $\to$ Generation $\to$ Synthesis** pipeline:

graph TD
    classDef input fill:#2d3436,stroke:#81ecec,stroke-width:2px,color:#fff;
    classDef encoder fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef latent fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff;
    classDef dit fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff;
    classDef synth fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff;

    A[Audio Input <br/> Real or Suno/Udio]:::input --> B(AE² Analytic Encoder):::encoder
    
    B -->|Equivariance| ZP(Pitch z_p):::latent
    B -->|Equivariance| ZR(Rhythm z_r):::latent
    B -->|Equivariance| ZT(Timbre z_t):::latent
    
    ZP --> D{HiFRA-T DiT <br/> Flow Matching}:::dit
    ZR --> D
    ZT --> D
    
    C[Conditions / Control] --> D
    
    D -->|FactorLock| RP(Refined z_p):::latent
    D -->|FactorLock| RR(Refined z_r):::latent
    D -->|FactorLock| RT(Refined z_t):::latent
    
    RP --> S[Neural Synth Bank]:::synth
    RR --> S
    RT --> S
    
    S --> V[Vocoder & Mixer]:::synth
    V --> O[Final Edited Audio]:::input

---

## 🚀 Quick Start: FactorLock Inference

Experience surgical music editing without any instruction fine-tuning or paired editing datasets.

### 1. Installation
git clone https://github.com/YourOrg/SHAYI.git
cd SHAYI
conda create -n shayi python=3.12
conda activate shayi
pip install -r part_1/requirements.txt
pip install -r part_2/requirements.txt

### 2. Download Weights
Download the pre-trained $AE^2$ and HiFRA-T checkpoints from [HuggingFace (Link TBA)](#) and place them in:
*   `part_1/runs/stage1_nocontent/ckpt_step00040000.pt`
*   `part_2/runs/dit_checkpoints_bf16/shayi_dit_final.pt`

### 3. Run FactorLock Editing
Take any input audio latent, apply an energy knob to deterministically modify a specific factor, and keep the rest strictly locked.

# Example: Double the rhythm density (groove) of a track
python run_inference.py --target rhythm --knob_scale 2.0

# Example: Shift the global harmonic structure (e.g., Emaj -> Cmaj)
python run_inference.py --target pitch --knob_scale 1.8
Generated audios will be saved in the `infer/` directory. 

---

## 📊 Reproducing Paper Experiments

We provide one-click scripts to reproduce the quantitative results (Decoupled Conservation) presented in the paper.

**Table 1: Disentanglement Probe ($R^2$ Matrix)**
Validates that our $AE^2$ manifold is physically orthogonal.
python run_table1_probe.py

**Table 2: FactorLock Objective Conservation (MSE)**
Proves that editing one target factor (e.g., Rhythm) yields massive target variance with near-zero off-target leakage.
python run_table2_mse.py

---

## 🗺️ Roadmap & Future Work

SHAYI V1 establishes the mathematical foundation for orthogonal generative manifolds. Our roadmap for V2 includes:
- [ ] **Vocal-Aware Factorization**: Adding a separate branch for linguistic content and vocal articulation.
- [ ] **ChatCut Integration**: Enabling surgical temporal masking (e.g., "Change the hi-hats only in bar 16").
- [ ] **Differentiable DAW Mixer**: Upgrading the simple synthesizer bank to a fully differentiable EQ/Compression chain.
- [ ] **VST Plugin Release**: Bringing FactorLock directly into Ableton Live and FL Studio.

---

## 📄 Citation

If you find SHAYI useful for your research or startup, please consider citing our work:

@article{shayi2026,
  title={SHAYI: A Self-Supervised Hierarchical Analytic Yielding Instructor for Controllable and Editable Music Generation},
  author={Anonymous Authors},
  journal={arXiv preprint},
  year={2026}
}

## 🤝 Acknowledgments
Special thanks to the open-source communities of `audiocraft` and `PESTO`. 

---
*Built with ❤️ by a team of producers and researchers who believe AI should assist, not replace, human creativity.*
# SHAYI
