# 可解耦音乐生成：Multi-head VAE Encodec 架构修改报告

## 目标

在仅修改模型架构的前提下，将 Encodec 改为 **多路 head（Pitch / Timbre / Rhythm）**，并将 latent 改为 **VAE 正态采样**；保证 Encoder 输出与 Decoder 输入为 **512 维**，与给定规格一致。

---

## 数据流（与规格对应）

1. **Backbone**：`[B, 1, 32000]` → 5 层卷积下采样 320 倍 → **Feat `[B, 128, 100]`**
2. **三路 Head + Adapter**：Feat(128) + 适配后辅助(64) → 每路输出 (μ, σ) → **Pitch 256-d, Timbre 128-d, Rhythm 128-d**
3. **Latent**：三路重参数化采样 → `z_pitch [B,256,100]`, `z_timbre [B,128,100]`, `z_rhythm [B,128,100]` → 通道维拼接 → **`[B, 512, 100]`**
4. **Decoder**：`[B, 512, 100]` → 5 层转置卷积上采样 → **重建波形 `[B, 1, 32000]`**

---

## 修改文件与内容

### 1. 新增：`audiocraft/audiocraft/modules/multihead_vae_encoder.py`

- **`Adapter1d`**：将辅助信号（f0 / timbre cluster / onsets）映射到 64 维并插值到目标时间步。
- **`DistributionHead`**：单路 head：`feat(128) + cond(64)` → 1×1 卷积 → 输出 (μ, log_var)，用于该路 latent 维。
- **`MultiheadVAEEncoder`**：
  - 使用现有 **SEANetEncoder** 作为共享 **Backbone**（`dimension=128`，`ratios=[8,5,4,2]`，hop_length=320），得到 `[B, 128, 100]`。
  - 三个 **Adapter** + 三个 **Head**：Pitch 256-d、Timbre 128-d、Rhythm 128-d，每路输出 (μ, log_var)。
  - **重参数化**：`z = μ + σ·ε`，ε～N(0,1)；三路在通道维拼接 → **输出 `[B, 512, 100]`**。
  - `forward(x, aux_pitch=None, aux_timbre=None, aux_rhythm=None)`：无辅助时用零向量作为 cond。
- **`build_multihead_vae_encoder(...)`**：根据 `sample_rate`、`channels`、`ratios` 等构建 Backbone + MultiheadVAEEncoder。

**常量**：`FEAT_DIM=128`，`ADAPTER_OUT_DIM=64`，`PITCH_LATENT_DIM=256`，`TIMBRE_LATENT_DIM=128`，`RHYTHM_LATENT_DIM=128`，`TOTAL_LATENT_DIM=512`。

---

### 2. 修改：`audiocraft/audiocraft/quantization/base.py`

- **新增 `ContinuousLatentQuantizer`**：
  - 用于 VAE 连续 latent，不做离散量化。
  - `forward(x, frame_rate)`：返回 `QuantizedResult(x=x, codes=x, bandwidth=..., penalty=None, metrics={})`。
  - `encode(x)` / `decode(codes)`：恒等，即 codes 即为连续 latent `[B, 512, T]`。
  - 实现 `total_codebooks=1`、`num_codebooks=1`、`bins`（cardinality，默认 65536）以兼容现有 `CompressionModel` 接口（如 `cardinality`、带宽计算等）。

---

### 3. 修改：`audiocraft/audiocraft/quantization/__init__.py`

- 在导出中增加 **`ContinuousLatentQuantizer`**，供 builders 使用。

---

### 4. 修改：`audiocraft/audiocraft/models/builders.py`

- **`get_quantizer`**：
  - 增加分支 `"continuous_latent"` → `qt.ContinuousLatentQuantizer`，并传入 `dimension`（以及可选的 `cardinality`）。
  - 当配置中无对应 quantizer 段时，`kwargs` 为空字典，仍能正确构造。

- **`get_encodec_autoencoder`**：
  - 增加分支 **`encoder_name == "multihead_vae"`**：
    - 使用 **`build_multihead_vae_encoder`** 构建 Encoder（Backbone 128-d + 3 heads + VAE 采样，输出 512-d）。
    - 使用 **SEANetDecoder**，**`dimension=512`**，`ratios` 与 backbone 一致（如 `[8,5,4,2]`），保证 `[B, 512, 100]` → `[B, 1, 32000]`。
  - 其他 seanet 参数（n_filters、n_residual_layers、causal、norm 等）从 `cfg.seanet` 读取并与默认一致。

这样，**Encoder 输出 512 维**、**Decoder 输入 512 维**，与规格一致；MusicGen 侧只需在配置中选用该 compression 模型，其输入维度即与 Encodec encoder 输出一致（512-d latent）。

---

### 5. 修改：`audiocraft/audiocraft/modules/__init__.py`

- 导出 **`MultiheadVAEEncoder`**、**`build_multihead_vae_encoder`**、**`TOTAL_LATENT_DIM`**，便于其他模块或脚本使用。

---

### 6. 新增：`config/model/encodec/encodec_multihead_vae.yaml`

- 定义 **multihead VAE Encodec** 的模型配置：
  - `encodec.autoencoder: multihead_vae`
  - `encodec.quantizer: continuous_latent`
  - `seanet` 与默认 encodec 对齐（`ratios: [8, 5, 4, 2]` 等），decoder 由 builder 固定为 **dimension=512**。
  - `continuous_latent.cardinality: 65536`（用于接口兼容）。

**使用方式**：在 solver/训练配置中指定模型为 `encodec/encodec_multihead_vae`（或通过 override 设置 `encodec.autoencoder=multihead_vae`、`encodec.quantizer=continuous_latent`），即可使用多路 VAE 架构，无需改训练脚本或损失定义。

---

## 接口与维度小结

| 环节           | 张量形状 / 说明 |
|----------------|------------------|
| 输入音频       | `[B, 1, 32000]`（1s @ 32kHz） |
| Backbone 输出  | `[B, 128, 100]`（Feat） |
| 每路 (μ, σ)    | Pitch `[B,256,100]`×2；Timbre/Rhythm 各 `[B,128,100]`×2 |
| Encoder 输出 z | `[B, 512, 100]`（Pitch\|Timbre\|Rhythm 通道拼接） |
| Decoder 输入   | `[B, 512, 100]` |
| 重建音频       | `[B, 1, 32000]` |

- **frame_rate**：由 backbone 的 hop_length 决定，当前为 32000/320 = **100 Hz**。
- **CompressionModel**：`channels` 仍为 1（音频通道）；连续 latent 的“维度”为 512，由 encoder/decoder 与 `ContinuousLatentQuantizer` 的 dimension 一致保证。

---

## 如何启用该架构

在训练或评估的 Hydra 配置中指定模型为 multihead VAE Encodec，例如：

```yaml
defaults:
  - /model: encodec/encodec_multihead_vae
```

或单独覆盖：

```yaml
model:
  encodec:
    autoencoder: multihead_vae
    quantizer: continuous_latent
```

并确保 `encodec.sample_rate`、`channels` 与数据集一致（如 32kHz、单声道）。Decoder 的 512 维输入由 builder 在 `multihead_vae` 分支内写死，无需在 yaml 中再写 dimension。
