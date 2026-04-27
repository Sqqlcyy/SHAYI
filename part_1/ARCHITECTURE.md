# SHAYI AE — Architecture & File Map

Three Stage-1 variants share the same structured factor heads (`z_p`,
`z_r`, `z_t`) but differ in how they produce those latents and how they
reconstruct audio:

| Variant | Encoder path | Decoder | Disentanglement story |
|---|---|---|---|
| **B-Codec** (mainline) | shared DAC codec trunk → per-head bottleneck | frozen DAC | soft (group-LASSO + aux-FiLM + HSIC) |
| **A** (feature-only) | 3 *independent* encoders (CQT / onset / mel) | from-scratch SEANet | hard (physical input separation) + soft |
| **A-Codec** (hybrid) | same 3 independent encoders | frozen DAC | hard (physical) + codec-distill |

Each variant maps cleanly onto an ablation axis. The file map at the bottom
lists the exact module boundaries.

---

## 1. Data Flow — Variant B-Codec

```
                                ┌── L_recon (MR-STFT + log-mel + wav L1)
                                ├── L_codec_distill = MSE(ĉ, c.detach())
                  Stage 1       ├── L_ortho (cross-cov | HSIC) on (z_p,z_r,z_t)
                  losses ──────>┤── L_trunk_L1 (group LASSO on trunk input conv)
                                ├── L_pitch_anchor (KL warmup)
                                └── (Stage 2 adds L_AY: 6 inv + 3 equiv + 3 comm)

 waveform x [B,1,T]
       │
       ▼
 ┌──────────────┐
 │ Frozen DAC   │  encoder (44.1 kHz)
 │   encoder    │
 └──────┬───────┘
        │ c [B, 1024, T_c=861]
        ├──────────┬──────────┐
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │Trunk_p │ │Trunk_r │ │Trunk_t │   per-head 3-layer Conv1d + GroupNorm + SiLU
   │ 1024   │ │ 1024   │ │ 1024   │   group-LASSO ⇒ each trunk reads only a sparse
   │ → 256  │ │ → 128  │ │ → 256  │   subset of the 1024 codec channels
   └───┬────┘ └───┬────┘ └───┬────┘
       │ aux_p     │ aux_r    │ aux_t   ← per-step Bernoulli(0.1) drop (Music ControlNet)
       │ CQT 588   │ onset 8  │ env 80  ← envelope = CheapTrick (pitch-invariant)
       ▼           ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │AuxFiLM │ │AuxFiLM │ │AuxFiLM │   (1+γ(aux))·trunk + β(aux), γ_init = 0.1
   └───┬────┘ └───┬────┘ └───┬────┘
       ▼          ▼          ▼
 ┌───────────┐ ┌────────┐ ┌──────────┐
 │ PitchHead │ │RhythmHd│ │TimbreHd  │
 │ Toeplitz  │ │ Conv   │ │ Global   │
 │ ×5 gain   │ │ stack  │ │ +LayerN  │
 │ +Softmax  │ │        │ │ +TimePool│
 └─────┬─────┘ └───┬────┘ └────┬─────┘
       ▼           ▼           ▼
   z_p [B,4,128,T]  z_r [B,64,T]  z_t [B,128]
       │           │           │
       ▼           │           │  (z_t never enters main path; only FiLM)
 ┌───────────┐    │           │
 │PitchEmbed │    │           │
 │ K·NB→128  │    │           │
 └─────┬─────┘    │           │
       │          │           │
       └────┬─────┘           │
            ▼                 │
    ┌──────────────┐          │
    │   concat     │          │
    │  [192, T]    │          │
    └──────┬───────┘          │
           ▼                  │
    ┌──────────────────┐      │
    │  FiLM Fusion     │◀─────┤  γ = tanh(Linear(z_t)) × 0.5
    │  4× Conv1d       │      │  z_t = 0 ⇒ γ = β = 0
    │  per-layer FiLM  │      │         ⇒ decoder sees no timbre
    │ [512,512,1024]   │      │         ⇒ z_t is load-bearing
    │ → codec_dim 1024 │      │
    └────────┬─────────┘      │
             ▼ ĉ [B,1024,T_c] │
    ┌───────────────┐         │
    │ Frozen DAC    │         │
    │   decoder     │         │
    └──────┬────────┘         │
           ▼                  ▼
        x̂ [B,1,T]        TimbreFiLMGen
                          (z_t → per-layer (γ, β), tanh-clamped)
```

---

## 2. Data Flow — Variant A 

Independent encoders per factor feature. No shared trunk. Reconstruction
via a from-scratch SEANet waveform decoder plus a MSD/MPD/MSSD adversarial
loss (HiFi-GAN / BigVGAN style). Optional envelope-FiLM aux inside the
content encoder, and GRL adversaries on z_t / z_r / z_content (**v2**).

```
 waveform x [B,1,T]                    ← raw audio (target only, not an input)
       │
   ┌───┴────────┬─────────────┐
   │            │             │
   ▼            ▼             ▼
 CQT 588     multi-onset   mel-spec 128  (aux_timbre = CheapTrick envelope 80d)
 feat_pitch  feat_rhythm   feat_mel
 [B,588,T]   [B,8,T]       [B,128,T]
   │            │             │
   ▼            ▼             ▼
┌───────┐    ┌───────┐    ┌────────┐
│Enc_p  │    │Enc_r  │    │Enc_c   │    TCN-style residual 1D convs;
│3 blks │    │3 blks │    │5 blks  │    no time downsampling —
│ 256   │    │ 256   │    │ 256    │    features already at 86 fps
└───┬───┘    └───┬───┘    └───┬────┘
    │            │            │
    │            │   (v2 opt) envelope-FiLM ← aux_timbre (80d, liftered)
    │            │            │
    ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌──────────┐
│PitchHead│ │RhythmHd │ │TimbreHd  │
│ Toeplitz│ │ Conv    │ │ Pool(T)  │
│ +Softmax│ │ stack   │ │ +LayerN  │
└────┬────┘ └────┬────┘ └─────┬────┘
     │           │            │
     ▼           ▼            ▼
   z_p        z_r          z_t                            c_content
 [B,4,128,T] [B,64,T]    [B,128]                          [B,256,T]  ← "content residual"
     │           │            │                              │
     ▼           │            │                              │
  PitchEmbed    │            │                              │
   → [B,128,T]  │            │                              │
     │          │            │                              │
     └─────┬────┘            │                              │
           ▼                 │                              │
     ┌───────────────┐       │                              │
     │ VariantAFusion│◀──────┤  γ = tanh(Linear(z_t))×0.5   │
     │ 2 convs + FiLM│       │                              │
     │ → [B, 256, T] │       │                              │
     └───────┬───────┘       │                              │
             │ + Proj(c_content) ←──────────────────────────┘ (residual carries
             ▼                                                phase / transients /
     ┌──────────────────┐                                     non-factor content)
     │ SEANetDecoder   │ ratios [8,8,4,2] = 512× upsample
     │ + AliasFree     │ BigVGAN Snake-β activation
     │   Snake-β       │ + 2-layer LSTM bottleneck
     │ + adv MPD+MSD+  │
     │   MSSD + EMA    │
     └────────┬────────┘
              ▼
         x̂ [B, 1, T_audio]
```

### v2 anti-leak additions (ae_variant_a_v2)

Five GRL adversaries + three always-on anchors address the leakage
patterns we measured on v1 (`z_p` dead, `z_t` leaks pitch+rhythm 0.5+,
`z_r` leaks pitch+envelope 0.6):

```
  z_t   ─GRL→ [pitch critic]   → MSE(·, pool(aux_pitch))    × w_adv_pitch_on_zt
  z_t   ─GRL→ [rhythm critic]  → MSE(·, pool(aux_rhythm))   × w_adv_rhythm_on_zt
  z_r   ─GRL→ [pitch critic]   → MSE(·, pool(aux_pitch))    × w_adv_pitch_on_zr
  z_r   ─GRL→ [envelope critic]→ MSE(·, pool(aux_timbre))   × w_adv_envelope_on_zr
  z_content (pool) ─GRL→ 3 critics (pitch/rhythm/envelope)  × w_content_adv

  plus: pitch_anchor + timbre_anchor + rhythm_anchor
        (always-on full strength — no decay schedule)
```

GRL is a variational *upper bound* on mutual information: minimising this
loss drives `MI(z, factor) → 0`. See `losses/grl_adv.py`.

---

## 3. Data Flow — Variant A-Codec (hybrid)

Variant A's independent encoders + Variant B's frozen DAC decoder. Encoder
path is physically separated (strong disentanglement); DAC decoder solves
the "train a vocoder from scratch" problem (good phase). The encoder has
to regress `c_hat ≈ c_target` in codec latent space — a harder objective
than Variant B's near-identity c → c, so convergence is slower.

```
 feat_pitch, feat_rhythm, feat_mel (+ envelope)
        │            │            │
        ▼            ▼            ▼
     Enc_p        Enc_r        Enc_c   (same as Variant A)
        │            │            │
        ▼            ▼            ▼
      z_p          z_r          z_t   +  c_content [B,256,T]
        │            │            │         │
        └────────────┴────────────┤         │
                                  ▼         │
                       ┌─── FiLMFusion ─────┤ (same FiLMFusion as B)
                       │   → [B, codec_dim=1024, T]
                       │         + Proj(c_content)
                       ▼
                  ĉ [B, 1024, T]
                       │
                       ├──► L_codec_distill = MSE(ĉ, DAC_enc(x).detach())
                       ▼
                ┌──────────────┐
                │ Frozen DAC   │
                │   decoder    │
                └──────┬───────┘
                       ▼
                    x̂ [B, 1, T_audio]
```

Content-residual adversary (3 GRL heads on pool(z_content)) prevents the
residual channel from hoarding factor information that the three explicit
heads should carry.

---

## 4. Latent shapes (10 s @ 44.1 kHz, hop=512, T=861)

| Latent | Shape | Type | Notes |
|---|---|---|---|
| `z_p` | `[B, 4, 128, 861]` | softmax(128 bins), K=4 polyphony | same across all variants |
| `z_r` | `[B, 64, 861]` | continuous, time-equivariant | same across all variants |
| `z_t` | `[B, 128]` | global (time-pooled) | same across all variants |
| `z_content` (A / A-Codec only) | `[B, 256, 861]` | residual content | carries phase / ambience / non-factor info |
| `ĉ` (B / A-Codec only) | `[B, 1024, 861]` | DAC codec latent | fed into frozen DAC decoder |

---

## 5. Disentanglement defense layers

| Variant | Input-side | Architecture | Output-side (v2) |
|---|---|---|---|
| **B** | envelope aux replaces MFCC, aux-dropout 0.1 | group-LASSO on trunk input conv, FiLM-only z_t injection, γ tanh-clamp | optional PitchAdversary on z_t |
| **A** / **A v2** | envelope FiLM inside content encoder (pitch-invariant input) | **physical** separation: pitch enc never sees rhythm, etc.; Toeplitz pitch head; time-pool z_t | GRL adversaries on z_t / z_r / z_content; always-on anchors |
| **A-Codec** | same as A v2 | same as A v2 + codec-distill pressure | same as A v2 |

Physical separation (independent encoder inputs) is a strictly stronger
disentanglement primitive than soft losses — `Enc_pitch` is **unable** to
encode rhythm because the onset stream isn't in its inputs. Soft losses
(HSIC, GRL) then handle the residual couplings that come from common
sources (mel contains pitch harmonics, etc.).

---

## 6. File Architecture

```
cloud_code/part_1/
├── ae_variant_b_codec/          ← Variant B-Codec (mainline)
│   ├── __init__.py
│   ├── codec_bridge.py          ← _CodecBridge (frozen DAC) + SeanetSmokeConfig
│   └── model.py                 ← VariantBCodec + VariantBCodecConfig + FiLMFusion
│
├── ae_variant_a/                ← Variant A (feature-only, part1_old lineage)
│   ├── __init__.py
│   └── model.py                 ← VariantAMultiHead + VariantAConfig + FeatureEncoder
│                                  (+ envelope FiLM, 5 GRL adversaries, SEANet decoder)
│
├── ae_variant_a_codec/          ← Variant A-Codec (hybrid)
│   ├── __init__.py
│   └── model.py                 ← VariantACodec + VariantACodecConfig
│                                  (A's encoders + B's FiLMFusion + frozen DAC)
│
├── models/                      ← shared neural building blocks
│   ├── trunks.py                ← per-head Trunk + group-LASSO penalty (B only)
│   ├── heads.py                 ← ToeplitzLinear / AuxFiLMInject / PitchHeadToeplitz /
│   │                              RhythmHeadConv / TimbreHeadGlobal / TimbreFiLMGen /
│   │                              PitchEmbed / RhythmScalarProjector /
│   │                              TimbreBrightnessProjector  (GRL re-export for compat)
│   ├── seanet_wrap.py           ← SEANet encoder/decoder wrapper (Variant A decoder)
│   └── _seanet_core/            ← vendored SEANet conv/lstm/seanet + Snake/SnakeBeta
│
├── losses/
│   ├── recon.py                 ← multi-res STFT + log-mel + wav L1
│   ├── ortho.py                 ← cross-cov + HSIC (z_p 4D canonicalize)
│   ├── leakage.py               ← linear_probe_r2 + mine_mi_nats (diagnostics)
│   ├── grl_adv.py               ← FactorAdversary + gradient_reverse (used by all variants)
│   ├── adv.py                   ← MPD + MSD + MSSD HiFi-GAN-style discriminators
│   ├── scalar_equiv.py          ← Stage 2 ratio losses (PESTO / Quinton / additive)
│   ├── kl.py                    ← gaussian KL (deterministic variants use w_kl=0)
│   ├── moe_aux.py               ← variance-of-energy aux (v2 uses 0)
│   └── __init__.py
│
├── transforms/                  ← Stage 2 equivariance transform group (7 ops)
│   ├── equivariant_group.py
│   ├── pair_sampling.py
│   └── __init__.py
│
├── stage1/                      ← Stage 1: AE training
│   ├── trainer.py               ← Stage1Trainer + anchors (pitch/timbre/rhythm)
│   │                              + leakage probe + MINE MI diagnostic + TB/wandb
│   ├── adv_trainer.py           ← Stage1AdvTrainer (HiFi-GAN adv + EMA + MSSD)
│   └── run.py                   ← entrypoint (yaml → build → fit) + variant validator
│
├── stage2/                      ← Stage 2: + L_AY
│   ├── trainer.py
│   ├── ay_loss_v2.py
│   └── run.py
│
├── stage3/                      ← Stage 3: AYI MLP identifiability probe
│   ├── ayi_mlp.py
│   ├── trainer.py
│   └── run.py
│
├── data/src_data_prep/          ← preprocessing (raw → processed wav → aux features)
│   ├── 00_download_*.py
│   ├── 01_preprocess_musdb18hq.py
│   ├── 01_preprocess_slakh.py
│   ├── 02_extract_pitch.py          ← CQT salience [588 bins]
│   ├── 03b_extract_rhythm_multi.py  ← 8-ch onset + tempogram (default)
│   ├── 04_extract_timbre.py         ← mel_linear + spec + mfcc
│   ├── 04b_extract_envelope.py      ← CheapTrick envelope [80 dim, pitch-invariant]
│   ├── 10_build_mini_dataset.py     ← train/val/test split listings
│   └── preprocess_parallel.py       ← parallel runner over 02/03b/04/04b
│
├── configs/
│   ├── default.yaml                 ← shared preprocessing settings
│   ├── train_b_film.yaml            ← Variant B Stage 1 (mainline)
│   ├── train_a_multihead.yaml       ← Variant A Stage 1 (v1)
│   ├── train_a_multihead_v2.yaml    ← Variant A Stage 1 (v2 — anti-leak, 30k steps)
│   ├── train_a_envaux.yaml          ← Variant A + envelope FiLM + timbre anchor
│   ├── train_a_codec.yaml           ← Variant A-Codec Stage 1 (25k steps)
│   ├── train_a_codec_v2.yaml        ← Variant A-Codec with output-side adversaries
│   ├── train_stage2.yaml            ← Stage 2 main config
│   └── train_stage3.yaml            ← Stage 3 main config
│
├── evaluation/                  ← all evaluation lives here (single source of truth)
│   ├── metrics.py               ← SNR / SI-SDR / log-STFT L1 / mel-dB L1 / FADComputer /
│   │                              ReconstructionMetricComputer
│   ├── probe.py                 ← unified disentanglement + recon probe CLI
│   │                              (leakage R² + MINE MI + recon + FAD + audio dump)
│   ├── reconstruct.py           ← reconstruction CLI (load ckpt → write wavs)
│   ├── dashboard.py             ← Streamlit dashboard
│   └── visualize.py             ← plotting helpers
│
├── inference/                   ← backwards-compat shims → evaluation.{dashboard,reconstruct}
├── utils/                       ← misc audio helpers
├── runtime.py                   ← build_model variant dispatch (A / A_CODEC / B_CODEC)
├── dataset.py                   ← Stage1AEDataset (audio + aux + feat loader)
└── requirements.txt
```

---

## 7. Loss formulae

### Stage 1 — all variants share the base

```
Stage 1 total =  w_recon · L_recon
              +  w_ortho · L_ortho(z_p, z_r, z_t)
              +  w_pitch_anchor(t) · KL(softmax(CQT_ds) ∥ mean_K(z_p))
```

### Variant B adds

```
              +  w_codec_distill · ‖ĉ − sg(c)‖²
              +  w_trunk_L1 · group_lasso(trunks_p/r/t input conv)
```

### Variant A v2 adds

```
              +  w_timbre_anchor · MSE(readout(z_t),       pool(envelope))
              +  w_rhythm_anchor · MSE(readout(pool(z_r)), pool(rhythm_multi))
              +  w_adv_pitch_on_zt     · GRL-MSE(critic(z_t),       pool(aux_pitch))
              +  w_adv_rhythm_on_zt    · GRL-MSE(critic(z_t),       pool(aux_rhythm))
              +  w_adv_pitch_on_zr     · GRL-MSE(critic(pool(z_r)), pool(aux_pitch))
              +  w_adv_envelope_on_zr  · GRL-MSE(critic(pool(z_r)), pool(aux_envelope))
              +  w_content_adv · Σ_f GRL-MSE(critic_f(pool(z_content)), pool(aux_f))
```

The 4+3 GRL terms collectively implement "MI(z, factor) → 0 for every
cross-factor pair". Anchors are kept full-strength throughout training in
v2 (no decay) after v1 showed z_p collapsing when anchors decayed at
step 5000.

### Variant A-Codec adds

Same as Variant A v2, minus the SEANet adversarial training, plus:

```
              +  w_codec_distill · ‖ĉ − sg(DAC_enc(x))‖²
```

### Stage 2 (all variants)

```
Stage 2 total =  Stage 1 total
              +  w_ay · ( c_inv(t)·Σ L_inv  +  c_eq(t)·Σ L_eq  +  c_comm(t)·Σ L_comm )

  L_eq_pitch  =  Huber( scalar(z_p_shift) / scalar(z_p_anchor)  −  2^(n/12) )      [PESTO]
  L_eq_rhythm =  Huber( scalar(z_r_i)     / scalar(z_r_j)       −  α_i / α_j )     [Quinton]
  L_eq_timbre =  Huber( (b_shift − b_anchor) − γ · Δdb )                            [additive]
  L_inv_*     =  MSE on z_non_target before/after non-target transform
  L_comm_*    =  MSE between (g_a∘g_b)·z and (g_b∘g_a)·z
```

### Stage 3

```
Stage 3      =  AYI MLP:  [Δz_p, Δz_r, Δz_t] → predict transform parameters Δa
              =  identifiability probe (frozen Stage 2 encoder)
```
