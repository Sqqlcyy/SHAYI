"""
音乐等变变换统一库 (Music Equivariant Transform Library)

包含三路特征的等变变换：
  - Pitch:   F0 (单音) 和 Pitch Salience (多音) 两套接口
  - Rhythm:  时间轴拉伸 / 平移
  - Timbre:  EQ tilt / 高切 / 共振峰缩放

设计原则：
  1. 只在离线特征层操作，不改动原始音频
  2. 每种变换只影响目标维度，不泄漏到其他维度
  3. 尽可能保持可微分（pitch salience transpose 可微，harmonic_reshape 因逐帧峰检测不可微）
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================
#  输入校验
# ============================================================

def _ensure_1d(x: Tensor, name: str) -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D [T], got shape={tuple(x.shape)}")


def _ensure_2d(x: Tensor, name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D [n_bins, T], got shape={tuple(x.shape)}")


# ============================================================
#  通用工具函数
# ============================================================

def _linear_resample_1d(x: Tensor, new_len: int) -> Tensor:
    if new_len <= 0:
        raise ValueError(f"new_len must be > 0, got {new_len}")
    if x.numel() == 0:
        return x
    y = F.interpolate(x.view(1, 1, -1), size=new_len, mode="linear", align_corners=False)
    return y.view(-1)


def _fractional_shift_freq_axis(salience: Tensor, shift_bins: float) -> Tensor:
    """
    沿频率轴做亚像素平移（可微分）。
    使用 grid_sample 实现，超出边界填零。
    
    Args:
        salience: [n_bins, T] 或 [B, n_bins, T]
        shift_bins: 平移的 bin 数（正 = 向高频移）
    """
    squeeze_batch = False
    if salience.ndim == 2:
        salience = salience.unsqueeze(0)
        squeeze_batch = True

    B, H, W = salience.shape
    x = salience.unsqueeze(1)  # [B, 1, H, W]

    grid_y = torch.linspace(-1.0, 1.0, H, device=salience.device, dtype=salience.dtype)
    grid_x = torch.linspace(-1.0, 1.0, W, device=salience.device, dtype=salience.dtype)

    freq_offset = shift_bins * (2.0 / H)
    grid_y_shifted = grid_y - freq_offset

    gy, gx = torch.meshgrid(grid_y_shifted, grid_x, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    out = out.squeeze(1)

    if squeeze_batch:
        out = out.squeeze(0)
    return out


def _detect_peaks_per_frame(
    salience: Tensor,
    threshold: float = 0.3,
    min_distance_bins: int = 3,
) -> Tuple[Tensor, Tensor]:
    """
    对每一帧检测 salience 峰值位置（局部极大值 + 阈值 + NMS）。
    
    Returns:
        peak_mask:   [n_bins, T] bool
        peak_values: [n_bins, T] float（峰处保留原值，其余为 0）
    """
    n_bins, T = salience.shape

    padded = F.pad(salience, (0, 0, 1, 1), mode="constant", value=0.0)
    is_local_max = (salience >= padded[:-2, :]) & (salience >= padded[2:, :])

    frame_max = salience.max(dim=0, keepdim=True).values.clamp(min=1e-8)
    above_threshold = salience > (threshold * frame_max)
    peak_mask = is_local_max & above_threshold

    # NMS：逐帧，从最强峰开始，抑制距离内的弱峰
    if min_distance_bins > 1:
        for t_idx in range(T):
            frame = salience[:, t_idx]
            mask_col = peak_mask[:, t_idx]
            if not mask_col.any():
                continue
            peak_indices = torch.where(mask_col)[0]
            peak_strengths = frame[peak_indices]
            sorted_order = peak_strengths.argsort(descending=True)
            kept = torch.ones(len(peak_indices), dtype=torch.bool, device=salience.device)
            for i in range(len(sorted_order)):
                idx_i = sorted_order[i]
                if not kept[idx_i]:
                    continue
                pos_i = peak_indices[idx_i]
                for j in range(i + 1, len(sorted_order)):
                    idx_j = sorted_order[j]
                    if not kept[idx_j]:
                        continue
                    if abs(int(peak_indices[idx_j]) - int(pos_i)) < min_distance_bins:
                        kept[idx_j] = False
            suppressed = peak_indices[~kept]
            peak_mask[suppressed, t_idx] = False

    peak_values = salience * peak_mask.float()
    return peak_mask, peak_values


# ============================================================
#  mel bin ↔ Hz 换算工具（用于 high_cut_eq 的 Hz 参数化）
# ============================================================

def _hz_to_mel(hz: float) -> float:
    """HTK mel scale"""
    return 2595.0 * torch.log10(torch.tensor(1.0 + hz / 700.0)).item()


def _mel_bin_for_hz(cutoff_hz: float, n_mels: int, fmin: float, fmax: float) -> int:
    """
    给定截止频率 (Hz)，返回对应的 mel bin 索引。
    假设 mel 滤波器组在 [fmin, fmax] 之间线性排列于 mel 刻度。
    """
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_cut = _hz_to_mel(cutoff_hz)

    # 截断到合法范围
    mel_cut = max(mel_min, min(mel_max, mel_cut))

    ratio = (mel_cut - mel_min) / (mel_max - mel_min + 1e-8)
    return int(round(ratio * (n_mels - 1)))


# ============================================================
#  主类：MusicEquivariantGroup
# ============================================================

class MusicEquivariantGroup:
    """
    音乐等变变换组。

    Args:
        bins_per_semitone: pitch salience 频率轴上每个半音占多少 bin。
            仅用于 salience 相关变换。默认 7.0 (对应 CQT bins_per_octave=84)。
        mel_fmin: mel 滤波器组最低频率 (Hz)，用于 high_cut_eq 的 Hz→bin 换算。
        mel_fmax: mel 滤波器组最高频率 (Hz)。
    """

    def __init__(
        self,
        bins_per_semitone: float = 7.0,
        mel_fmin: float = 20.0,
        mel_fmax: float = 8000.0,
    ):
        if bins_per_semitone <= 0:
            raise ValueError(f"bins_per_semitone must be > 0, got {bins_per_semitone}")
        self.bins_per_semitone = bins_per_semitone
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

    # ============================================================
    #  Pitch 分支 — F0 (单声道)
    # ============================================================

    def pitch_shift(self, f0: Tensor, semitone: float) -> Tensor:
        """F0 全局变调。voiced 帧乘以 2^(semitone/12)，unvoiced 帧不动。"""
        _ensure_1d(f0, "f0")
        scale = float(2.0 ** (float(semitone) / 12.0))
        voiced = f0 > 0
        out = f0.clone()
        out[voiced] = out[voiced] * scale
        return out

    def pitch_vibrato(
        self, f0: Tensor, amount: float, lfo_hz: float = 5.0, fps: float = 100.0
    ) -> Tensor:
        """给 F0 叠加正弦 vibrato。amount 单位为半音振幅。"""
        _ensure_1d(f0, "f0")
        if f0.numel() == 0 or float(amount) == 0.0:
            return f0.clone()

        duration_sec = f0.shape[0] / float(fps)
        t = torch.linspace(0.0, duration_sec, steps=f0.shape[0], device=f0.device, dtype=f0.dtype)
        lfo = torch.sin(2.0 * torch.pi * float(lfo_hz) * t)
        semitone_offset = float(amount) * lfo
        ratio = torch.pow(torch.tensor(2.0, device=f0.device, dtype=f0.dtype), semitone_offset / 12.0)

        out = f0.clone()
        voiced = out > 0
        out[voiced] = out[voiced] * ratio[voiced]
        return out

    # ============================================================
    #  Pitch 分支 — Pitch Salience (多声道/伴奏)
    # ============================================================

    def salience_transpose(self, salience: Tensor, semitone: float) -> Tensor:
        """
        Pitch Salience 全局变调：所有峰沿频率轴平移相同半音数。
        和弦结构（音程关系）完全保留，只改变调性。
        可微分（基于 grid_sample）。
        
        Args:
            salience: [n_bins, T]
            semitone: 半音偏移量（支持小数）
        """
        _ensure_2d(salience, "salience")
        if semitone == 0.0:
            return salience.clone()
        shift_bins = semitone * self.bins_per_semitone
        return _fractional_shift_freq_axis(salience, shift_bins)

    def salience_harmonic_reshape(
        self,
        salience: Tensor,
        interval_semitones: float,
        anchor: str = "lowest",
        peak_threshold: float = 0.3,
    ) -> Tensor:
        """
        谐波结构重塑：保持锚定音不动，改变其他声部与锚定音的音程。
        
        与 transpose 的区别：
          - transpose: 所有音平移相同量，音程不变
          - harmonic_reshape: 锚定音不动，和弦内部结构改变
        
        物理意义（anchor="lowest"）:
          - interval_semitones = +1:  C-E-G → C-F-G#（扩张）
          - interval_semitones = -1:  C-E-G → C-Eb-Gb（收缩）
        
        注意：因逐帧峰检测，此操作不可微分。
        
        Args:
            salience: [n_bins, T]
            interval_semitones: 音程缩放量（正=扩张，负=收缩）
            anchor: "lowest" | "highest" | "center"
            peak_threshold: 峰检测相对阈值
        """
        _ensure_2d(salience, "salience")
        if interval_semitones == 0.0:
            return salience.clone()

        n_bins, T = salience.shape
        bps = self.bins_per_semitone

        peak_mask, _ = _detect_peaks_per_frame(salience, threshold=peak_threshold)

        # 背景（非峰部分）直接保留
        out = salience * (~peak_mask).float()

        sigma = bps * 0.3  # 高斯核宽度 ≈ 半音的 30%
        bin_indices = torch.arange(n_bins, device=salience.device, dtype=salience.dtype)

        for t_idx in range(T):
            frame_peaks = peak_mask[:, t_idx]
            if not frame_peaks.any():
                continue

            peak_positions = torch.where(frame_peaks)[0].float()
            peak_vals = salience[frame_peaks, t_idx]

            # 锚定位置
            if anchor == "lowest":
                anchor_pos = peak_positions.min()
            elif anchor == "highest":
                anchor_pos = peak_positions.max()
            elif anchor == "center":
                anchor_pos = (peak_positions * peak_vals).sum() / peak_vals.sum().clamp(min=1e-8)
            else:
                raise ValueError(f"Unknown anchor: {anchor}")

            offsets = peak_positions - anchor_pos
            direction = torch.sign(offsets)
            new_positions = anchor_pos + offsets + direction * interval_semitones * bps

            for pk_idx in range(len(new_positions)):
                pos = new_positions[pk_idx]
                val = peak_vals[pk_idx]
                if pos < 0 or pos >= n_bins:
                    continue
                gaussian = torch.exp(-0.5 * ((bin_indices - pos) / sigma) ** 2)
                gaussian = gaussian / gaussian.max().clamp(min=1e-8)
                out[:, t_idx] = out[:, t_idx] + val * gaussian

        return out.clamp(min=0.0)

    # ============================================================
    #  Rhythm 分支
    # ============================================================

    def rhythm_stretch(self, rhythm_map: Tensor, scale: float) -> Tensor:
        """节奏拉伸/压缩。scale>1 变慢（变长），scale<1 变快（变短）。"""
        _ensure_1d(rhythm_map, "rhythm_map")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        T = rhythm_map.shape[0]
        if T == 0 or scale == 1.0:
            return rhythm_map.clone()
        stretched_len = max(1, int(round(T * float(scale))))
        return _linear_resample_1d(rhythm_map, stretched_len)

    def rhythm_shift(self, rhythm_map: Tensor, shift_steps: int) -> Tensor:
        """节奏平移。正值在前方补零（延迟），负值截掉前方帧（提前）。"""
        _ensure_1d(rhythm_map, "rhythm_map")
        T = rhythm_map.shape[0]
        s = int(shift_steps)
        if T == 0 or s == 0:
            return rhythm_map.clone()
        if s > 0:
            zeros = torch.zeros(s, device=rhythm_map.device, dtype=rhythm_map.dtype)
            return torch.cat([zeros, rhythm_map], dim=0)
        else:
            s_abs = abs(s)
            if s_abs >= T:
                return torch.zeros(0, device=rhythm_map.device, dtype=rhythm_map.dtype)
            return rhythm_map[s_abs:]

    # ============================================================
    #  Timbre 分支
    # ============================================================

    def timbre_high_cut_eq(
        self,
        mel_spec: Tensor,
        cutoff_hz: float,
    ) -> Tensor:
        """
        高切 EQ：以 cutoff_hz 为截止频率，其上方做余弦衰减。
        
        截止频率通过 mel 刻度换算到 bin 索引，因此与你的 mel 滤波器组参数
        (mel_fmin, mel_fmax) 物理对应。
        
        Args:
            mel_spec:  [n_mels, T]
            cutoff_hz: 截止频率 (Hz)。
                       例如 4000.0 表示 4kHz 以上开始衰减。
                       如果 >= mel_fmax 则不做任何操作（等价于不开启）。
                       如果 <= mel_fmin 则几乎全部衰减。
        """
        _ensure_2d(mel_spec, "mel_spec")

        n_mels = mel_spec.shape[0]

        # Hz → mel bin
        cut_bin = _mel_bin_for_hz(cutoff_hz, n_mels, self.mel_fmin, self.mel_fmax)

        # 如果截止 bin 已经在最后或超出，不做操作
        if cut_bin >= n_mels - 1:
            return mel_spec.clone()

        # 截止 bin 在最前面的极端情况
        cut_start = max(0, cut_bin)

        out = mel_spec.clone()
        decay_length = n_mels - cut_start
        if decay_length <= 0:
            return out

        # 余弦衰减：从 1.0 平滑到底线增益 0.1
        t = torch.linspace(0.0, torch.pi / 2.0, steps=decay_length, device=mel_spec.device, dtype=mel_spec.dtype)
        decay_curve = torch.cos(t)
        decay_curve = 0.1 + 0.9 * decay_curve

        out[cut_start:, :] = out[cut_start:, :] * decay_curve.unsqueeze(1)
        return out

    def timbre_eq_tilt(self, mel_spec: Tensor, tilt_db: float) -> Tensor:
        """频谱倾斜：正值提亮高频/压暗低频，负值反之。"""
        _ensure_2d(mel_spec, "mel_spec")
        n_mels = mel_spec.shape[0]
        if n_mels < 2 or tilt_db == 0.0:
            return mel_spec.clone()

        out = mel_spec.clone()
        gains = torch.linspace(
            -float(tilt_db) / 2.0, float(tilt_db) / 2.0,
            steps=n_mels, device=mel_spec.device, dtype=mel_spec.dtype,
        )
        gains_linear = torch.pow(
            torch.tensor(10.0, device=mel_spec.device, dtype=mel_spec.dtype), gains / 20.0
        )
        out = out * gains_linear.unsqueeze(1)
        return out

    def timbre_formant(self, mel_spec: Tensor, formant_scale: float) -> Tensor:
        """
        共振峰缩放：频率轴拉伸/压缩，模拟声道长度变化。
        formant_scale > 1 升高共振峰，< 1 降低。
        
        修复：formant_scale < 1 时高频区域做余弦衰减而非硬填零。
        """
        _ensure_2d(mel_spec, "mel_spec")
        orig_shape = mel_spec.shape
        if orig_shape[0] < 2 or orig_shape[1] < 2 or formant_scale == 1.0:
            return mel_spec.clone()

        x = mel_spec.unsqueeze(0).unsqueeze(0)
        new_mels = max(2, int(round(orig_shape[0] * formant_scale)))
        x_scaled = F.interpolate(x, size=(new_mels, orig_shape[1]), mode='bilinear', align_corners=False)
        scaled_mel = x_scaled.squeeze(0).squeeze(0)

        n_mels_orig = orig_shape[0]
        if new_mels >= n_mels_orig:
            out = scaled_mel[:n_mels_orig, :]
        else:
            out = torch.zeros(n_mels_orig, orig_shape[1], device=mel_spec.device, dtype=mel_spec.dtype)
            out[:new_mels, :] = scaled_mel
            # 余弦衰减填充，避免硬截断
            tail_value = scaled_mel[-1:, :]
            remaining = n_mels_orig - new_mels
            if remaining > 0:
                decay = 0.5 * (1.0 + torch.cos(
                    torch.pi * torch.linspace(0.0, 1.0, steps=remaining,
                                              device=mel_spec.device, dtype=mel_spec.dtype)
                ))
                out[new_mels:, :] = tail_value * decay.unsqueeze(1)
        return out

    # ============================================================
    #  统一调用接口
    # ============================================================

    def __call__(
        self,
        f0: Tensor,
        rhythm_map: Tensor,
        mel_spec: Tensor,
        salience: Tensor,
        fps: float = 100.0,
        # --- Pitch (F0) ---
        semitone: float = 0.0,
        vibrato_amount: float = 0.0,
        vibrato_lfo_hz: float = 5.0,
        # --- Pitch (Salience) ---
        salience_semitone: float = 0.0,     # 默认 0.0 → 恒等
        salience_interval: float = 0.0,     # 默认 0.0 → 恒等
        salience_anchor: str = "lowest",
        salience_peak_threshold: float = 0.3,
        # --- Rhythm ---
        stretch_scale: float = 1.0,
        rhythm_shift_steps: int = 0,
        # --- Timbre ---
        high_cut_hz: float = 20000.0,
        eq_tilt_db: float = 0.0,
        formant_scale: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        统一接口：对四路特征执行等变变换。
        所有变换无条件执行，靠默认参数值退化为恒等。

        Returns:
            (c_pitch, c_rhythm, c_timbre, c_salience)
        """

        # ---- Pitch (F0) ----
        c_p = self.pitch_shift(f0, semitone)
        c_p = self.pitch_vibrato(c_p, amount=vibrato_amount, lfo_hz=vibrato_lfo_hz, fps=fps)

        # ---- Pitch (Salience) ----
        c_s = self.salience_transpose(salience, salience_semitone)
        c_s = self.salience_harmonic_reshape(
            c_s,
            interval_semitones=salience_interval,
            anchor=salience_anchor,
            peak_threshold=salience_peak_threshold,
        )

        # ---- Rhythm ----
        c_r = self.rhythm_stretch(rhythm_map, stretch_scale)
        c_r = self.rhythm_shift(c_r, shift_steps=rhythm_shift_steps)

        # ---- Timbre ----
        c_t = self.timbre_high_cut_eq(mel_spec, cutoff_hz=high_cut_hz)
        c_t = self.timbre_eq_tilt(c_t, tilt_db=eq_tilt_db)
        c_t = self.timbre_formant(c_t, formant_scale=formant_scale)

        return c_p, c_r, c_t, c_s