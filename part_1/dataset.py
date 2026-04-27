from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.audio import load_mono_audio, peak_normalize, random_crop_1d


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Stage1DataConfig:
    roots: List[Path] = field(default_factory=list)      # e.g. [processed/slakh, processed/moisesdb, processed/musdb7s]
    track_list: Optional[Path] = None                    # optional train.txt / val.txt; one sample dir per line
    sample_rate: int = 44100
    crop_seconds: float = 10.0
    feat_hop: int = 512                                  # must match preprocessing (stft.hop_length)
    latent_hop: int = 512                                # codec/adapter latent hop for B_CODEC
    normalize_audio: bool = True
    augment: bool = False                                # polarity/gain randomization
    wav_name: str = "instru.wav"
    pitch_feat_name: str = "pitch_salience_instru_nondrum.npy"
    rhythm_feat_name: str = "rhythm_instru.npy"
    # Optional multi-channel rhythm aux (onset_env + tempogram + ...), used by
    # Variant B's rhythm adapter when a single scalar/frame is too thin to
    # fight timbre leakage. Produced by ``03b_extract_rhythm_multi.py``.
    rhythm_multi_feat_name: str = "rhythm_multi_instru.npy"
    timbre_feat_name: str = "mfcc_instru.npy"
    spec_feat_name: str = "spec_instru.npy"
    mel_linear_feat_name: str = "mel_linear_instru.npy"
    # Fallbacks for older preprocessing dumps. Pitch has NO fallbacks —
    # the 2-D CQT salience is load-bearing for Stage 2 equivariance,
    # and silently substituting a 1-D f0 would break the T_p aux shift.
    pitch_fallbacks: Tuple[str, ...] = ()
    rhythm_fallbacks: Tuple[str, ...] = ("rhythm.npy",)
    rhythm_multi_fallbacks: Tuple[str, ...] = ()
    timbre_fallbacks: Tuple[str, ...] = ("timbre_instru.npy", "timbre.npy")
    # Optional envelope-based pitch-invariant timbre aux (CheapTrick / true
    # envelope), produced by ``04b_extract_envelope.py``. When present and
    # ``prefer_envelope_timbre=True``, this replaces MFCC in ``aux_timbre``.
    envelope_feat_name: str = "envelope_instru.npy"
    envelope_fallbacks: Tuple[str, ...] = ()
    prefer_envelope_timbre: bool = False
    spec_fallbacks: Tuple[str, ...] = ("spec.npy",)
    mel_linear_fallbacks: Tuple[str, ...] = ("mel_linear.npy",)
    # Number of bins in the CQT pitch salience map. Must match
    # ``cqt.bins_per_octave * cqt.n_octaves`` in configs/default.yaml and
    # ``VariantBCodecConfig.aux_pitch_dim``.
    n_pitch_bins: int = 588
    n_mels: int = 128
    n_mfcc: int = 20
    # Channel count for the rhythm aux stream. 1 = legacy scalar onset map;
    # >1 = multi-channel (onset_env, tempogram bands, ...). The dataset
    # prefers ``rhythm_multi_feat_name`` when this is >1, falling back to
    # broadcasting the legacy 1-D file across channels if allowed.
    n_rhythm_channels: int = 1
    # Dim of the envelope timbre aux when ``prefer_envelope_timbre=True``.
    n_envelope_dim: int = 80
    # If True, fall back to a zero tensor when a pitch file is missing.
    # Default False because we got burned by a silent-flatten bug before —
    # production runs should fail loudly, smoke tests may flip this.
    allow_missing_aux: bool = False
    seed: int = 0

    @property
    def crop_samples(self) -> int:
        return int(round(self.sample_rate * self.crop_seconds))

    @property
    def latent_length(self) -> int:
        return self.crop_samples // self.latent_hop

    @property
    def feat_length(self) -> int:
        return self.crop_samples // self.feat_hop


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------
def _is_sample_dir(p: Path, wav_name: str) -> bool:
    return (p / wav_name).exists()


def _walk_sample_dirs(root: Path, wav_name: str) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for track_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if _is_sample_dir(track_dir, wav_name):
            out.append(track_dir)
            continue
        for seg in sorted([p for p in track_dir.iterdir() if p.is_dir()]):
            if _is_sample_dir(seg, wav_name):
                out.append(seg)
    return out


def _resolve_feat(d: Path, primary: str, fallbacks: Tuple[str, ...]) -> Optional[Path]:
    p = d / primary
    if p.exists():
        return p
    for fb in fallbacks:
        q = d / fb
        if q.exists():
            return q
    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Stage1AEDataset(Dataset):

    def __init__(self, cfg: Stage1DataConfig):
        self.cfg = cfg
        self.sample_dirs: List[Path] = []

        if cfg.track_list is not None and cfg.track_list.exists():
            with open(cfg.track_list, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    p = Path(line)
                    if _is_sample_dir(p, cfg.wav_name):
                        self.sample_dirs.append(p)
        else:
            for root in cfg.roots:
                self.sample_dirs.extend(_walk_sample_dirs(Path(root), cfg.wav_name))

        if len(self.sample_dirs) == 0:
            raise RuntimeError(
                f"Stage1AEDataset: no sample dirs found. "
                f"roots={cfg.roots}, track_list={cfg.track_list}"
            )

        # Pre-resolve feature paths so __getitem__ is cheap.
        self._feat_cache: Dict[int, Dict[str, Optional[Path]]] = {}
        missing_pitch: List[Path] = []
        for i, d in enumerate(self.sample_dirs):
            pitch_path = _resolve_feat(d, cfg.pitch_feat_name, cfg.pitch_fallbacks)
            if pitch_path is None:
                missing_pitch.append(d)
            self._feat_cache[i] = {
                "pitch": pitch_path,
                "rhythm": _resolve_feat(d, cfg.rhythm_feat_name, cfg.rhythm_fallbacks),
                "rhythm_multi": _resolve_feat(d, cfg.rhythm_multi_feat_name, cfg.rhythm_multi_fallbacks),
                "timbre": _resolve_feat(d, cfg.timbre_feat_name, cfg.timbre_fallbacks),
                "envelope": _resolve_feat(d, cfg.envelope_feat_name, cfg.envelope_fallbacks),
                "spec": _resolve_feat(d, cfg.spec_feat_name, cfg.spec_fallbacks),
                "mel_linear": _resolve_feat(d, cfg.mel_linear_feat_name, cfg.mel_linear_fallbacks),
            }

        if missing_pitch and not cfg.allow_missing_aux:
            sample = "\n  ".join(str(p) for p in missing_pitch[:5])
            raise RuntimeError(
                f"Stage1AEDataset: {len(missing_pitch)}/{len(self.sample_dirs)} "
                f"sample dirs are missing the pitch file "
                f"'{cfg.pitch_feat_name}'. First few:\n  {sample}\n"
                f"Re-run data/src_data_prep/02_extract_pitch.py to generate the "
                f"2-D CQT salience, or set Stage1DataConfig.allow_missing_aux=True "
                f"for a smoke/debug run (will emit all-zero pitch aux)."
            )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    # ------------------------------------------------------------------
    def _resample_feat_to_latent_T(self, feat: np.ndarray, start_sample: int) -> np.ndarray:
        feat_hop = self.cfg.feat_hop
        T_feat = feat.shape[0]
        # Feature index window corresponding to the audio crop
        feat_start = start_sample // feat_hop
        feat_end = feat_start + (self.cfg.crop_samples // feat_hop)
        feat_end = min(feat_end, T_feat)
        window = feat[feat_start:feat_end]
        if window.size == 0:
            window = np.zeros(1, dtype=np.float32)
        # Resample to latent frame rate via 1-D linear interpolation.
        T_lat = self.cfg.latent_length
        x = torch.from_numpy(window.astype(np.float32))[None, None, :]  # [1,1,T]
        y = F.interpolate(x, size=T_lat, mode="linear", align_corners=False)
        return y.squeeze(0).squeeze(0).numpy()

    def _crop_feature_1d(self, feat: np.ndarray, start_sample: int) -> np.ndarray:
        target_T = self.cfg.feat_length
        feat_start = start_sample // self.cfg.feat_hop
        feat_end = feat_start + target_T
        window = feat[feat_start:min(feat_end, feat.shape[0])]
        out = np.zeros(target_T, dtype=np.float32)
        if window.size > 0:
            out[:window.shape[0]] = window.astype(np.float32)
        return out

    def _crop_feature_2d(
        self,
        feat: np.ndarray,
        start_sample: int,
        *,
        expected_channels: int,
        name: str,
    ) -> np.ndarray:
        if feat.ndim != 2:
            raise RuntimeError(f"{name}: expected [C, T_feat], got shape {feat.shape}")
        if feat.shape[0] != expected_channels:
            raise RuntimeError(
                f"{name}: channel mismatch — file has {feat.shape[0]}, "
                f"expected {expected_channels}."
            )
        target_T = self.cfg.feat_length
        feat_start = start_sample // self.cfg.feat_hop
        feat_end = feat_start + target_T
        window = feat[:, feat_start:min(feat_end, feat.shape[1])]
        out = np.zeros((expected_channels, target_T), dtype=np.float32)
        if window.shape[1] > 0:
            out[:, :window.shape[1]] = window.astype(np.float32)
        return out

    def _load_pitch_salience(self, path: Optional[Path], start_sample: int) -> np.ndarray:

        cfg = self.cfg
        T_lat = cfg.latent_length
        if path is None:
            return np.zeros((cfg.n_pitch_bins, T_lat), dtype=np.float32)

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            raise RuntimeError(
                f"{path}: expected 2-D pitch salience [n_bins, T_feat], got shape {arr.shape}. "
                f"Re-run data/src_data_prep/02_extract_pitch.py."
            )
        if arr.shape[0] != cfg.n_pitch_bins:
            raise RuntimeError(
                f"{path}: pitch bin count mismatch — file has {arr.shape[0]} bins, "
                f"Stage1DataConfig.n_pitch_bins={cfg.n_pitch_bins}. "
                f"Re-extract pitch or update the config."
            )

        feat_hop = cfg.feat_hop
        T_feat = arr.shape[1]
        feat_start = start_sample // feat_hop
        feat_end = feat_start + (cfg.crop_samples // feat_hop)
        feat_end = min(feat_end, T_feat)
        window = arr[:, feat_start:feat_end]
        if window.shape[1] == 0:
            window = np.zeros((cfg.n_pitch_bins, 1), dtype=np.float32)

        x = torch.from_numpy(window)[None, :, :]  # [1, n_bins, T_crop]
        y = F.interpolate(x, size=T_lat, mode="linear", align_corners=False)
        return y.squeeze(0).numpy()  # [n_bins, T_lat]

    def _load_pitch_salience_feature_T(self, path: Optional[Path], start_sample: int) -> np.ndarray:
        cfg = self.cfg
        if path is None:
            return np.zeros((cfg.n_pitch_bins, cfg.feat_length), dtype=np.float32)
        arr = np.load(path).astype(np.float32)
        return self._crop_feature_2d(
            arr,
            start_sample,
            expected_channels=cfg.n_pitch_bins,
            name=str(path),
        )

    def _load_mel_feature_T(self, path: Optional[Path], start_sample: int) -> np.ndarray:
        cfg = self.cfg
        if path is None:
            if cfg.allow_missing_aux:
                return np.zeros((cfg.n_mels, cfg.feat_length), dtype=np.float32)
            raise RuntimeError(
                "Stage1AEDataset: missing mel/spec feature for Variant A. "
                f"Expected '{cfg.spec_feat_name}' or fallback {cfg.spec_fallbacks}."
            )
        arr = np.load(path).astype(np.float32)
        return self._crop_feature_2d(
            arr,
            start_sample,
            expected_channels=cfg.n_mels,
            name=str(path),
        )

    def _resample_2d_to_latent_T(self, arr: np.ndarray, start_sample: int) -> np.ndarray:
        cfg = self.cfg
        feat_hop = cfg.feat_hop
        T_feat = arr.shape[1]
        feat_start = start_sample // feat_hop
        feat_end = min(feat_start + (cfg.crop_samples // feat_hop), T_feat)
        window = arr[:, feat_start:feat_end]
        if window.shape[1] == 0:
            window = np.zeros((arr.shape[0], 1), dtype=np.float32)
        x = torch.from_numpy(window.astype(np.float32))[None, :, :]
        y = F.interpolate(x, size=cfg.latent_length, mode="linear", align_corners=False)
        return y.squeeze(0).numpy()

    def _load_rhythm_aux(
        self,
        rhythm_path: Optional[Path],
        rhythm_multi_path: Optional[Path],
        start_sample: int,
    ) -> np.ndarray:
        """Rhythm aux at latent rate, shape ``[n_rhythm_channels, T_lat]``.

        Preference order when ``n_rhythm_channels > 1``:
          1. multi-channel file (``rhythm_multi_feat_name``)
          2. legacy 1-D file broadcast into all channels (only under
             ``allow_missing_aux``, since it throws away the whole point
             of the multi-channel stream)
          3. zeros (smoke mode)
        """
        cfg = self.cfg
        n_ch = cfg.n_rhythm_channels
        T_lat = cfg.latent_length

        if n_ch <= 1:
            if rhythm_path is None:
                return np.zeros((1, T_lat), dtype=np.float32)
            arr = np.load(rhythm_path).astype(np.float32).reshape(-1)
            return self._resample_feat_to_latent_T(arr, start_sample)[None, :]

        if rhythm_multi_path is not None:
            arr = np.load(rhythm_multi_path).astype(np.float32)
            if arr.ndim != 2:
                raise RuntimeError(
                    f"{rhythm_multi_path}: expected [C, T_feat], got shape {arr.shape}"
                )
            if arr.shape[0] != n_ch:
                raise RuntimeError(
                    f"{rhythm_multi_path}: channel mismatch — file has "
                    f"{arr.shape[0]}, Stage1DataConfig.n_rhythm_channels={n_ch}."
                )
            return self._resample_2d_to_latent_T(arr, start_sample)

        if rhythm_path is not None and cfg.allow_missing_aux:
            arr = np.load(rhythm_path).astype(np.float32).reshape(-1)
            lat = self._resample_feat_to_latent_T(arr, start_sample)
            return np.broadcast_to(lat[None, :], (n_ch, T_lat)).copy()

        if cfg.allow_missing_aux:
            return np.zeros((n_ch, T_lat), dtype=np.float32)

        raise RuntimeError(
            f"Stage1AEDataset: n_rhythm_channels={n_ch} but no multi-channel "
            f"rhythm file found. Run data/src_data_prep/03b_extract_rhythm_multi.py "
            f"to generate '{cfg.rhythm_multi_feat_name}', or set "
            f"allow_missing_aux=True for a smoke run."
        )

    def _load_envelope_aux(self, path: Optional[Path], start_sample: int) -> np.ndarray:
        cfg = self.cfg
        if path is None:
            if cfg.allow_missing_aux:
                return np.zeros((cfg.n_envelope_dim, cfg.latent_length), dtype=np.float32)
            raise RuntimeError(
                "Stage1AEDataset: prefer_envelope_timbre=True but envelope "
                f"file '{cfg.envelope_feat_name}' not found. Run "
                "data/src_data_prep/04b_extract_envelope.py."
            )
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] != cfg.n_envelope_dim:
            raise RuntimeError(
                f"{path}: expected [{cfg.n_envelope_dim}, T_feat], got {arr.shape}"
            )
        return self._resample_2d_to_latent_T(arr, start_sample)

    def _load_timbre_aux(self, path: Optional[Path], start_sample: int) -> np.ndarray:

        cfg = self.cfg
        if path is None:
            return np.zeros((cfg.n_mfcc, cfg.latent_length), dtype=np.float32)

        arr = np.load(path).astype(np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != cfg.n_mfcc:
                raise RuntimeError(
                    f"{path}: global timbre dim mismatch — file has {arr.shape[0]}, "
                    f"expected {cfg.n_mfcc}."
                )
            return np.repeat(arr[:, None], cfg.latent_length, axis=1).astype(np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"{path}: expected MFCC [n_mfcc, T_feat] or global [n_mfcc], got {arr.shape}")
        if arr.shape[0] != cfg.n_mfcc:
            raise RuntimeError(
                f"{path}: MFCC dim mismatch — file has {arr.shape[0]}, expected {cfg.n_mfcc}."
            )

        feat_start = start_sample // cfg.feat_hop
        feat_end = min(feat_start + cfg.feat_length, arr.shape[1])
        window = arr[:, feat_start:feat_end]
        if window.shape[1] == 0:
            window = np.zeros((cfg.n_mfcc, 1), dtype=np.float32)
        x = torch.from_numpy(window)[None, :, :]
        y = F.interpolate(x, size=cfg.latent_length, mode="linear", align_corners=False)
        return y.squeeze(0).numpy()

    def __getitem__(self, idx: int) -> Dict:
        d = self.sample_dirs[idx]
        cfg = self.cfg

        wav_path = d / cfg.wav_name
        y, sr = load_mono_audio(wav_path, target_sr=cfg.sample_rate)
        if sr != cfg.sample_rate:
            raise RuntimeError(
                f"Sample rate mismatch at {wav_path}: got {sr}, expected {cfg.sample_rate}. "
                f"Re-run preprocessing with target_sr={cfg.sample_rate}."
            )
        if cfg.normalize_audio:
            y = peak_normalize(y)

        crop_samples = cfg.crop_samples

        # Random crop start — we also need it to align aux features
        n = y.shape[0]
        if n <= crop_samples:
            start_sample = 0
            audio_crop = np.zeros(crop_samples, dtype=np.float32)
            audio_crop[:n] = y
        else:
            start_sample = int(np.random.randint(0, n - crop_samples + 1))
            audio_crop = y[start_sample : start_sample + crop_samples].astype(np.float32)

        # Simple waveform augmentations (Stage 1 only; disabled in val)
        if cfg.augment:
            if np.random.rand() < 0.5:
                audio_crop = -audio_crop  # polarity flip
            gain = float(np.random.uniform(0.7, 1.0))
            audio_crop = audio_crop * gain

        # Aux features
        feats = self._feat_cache[idx]

        def _load_per_frame_1d(path: Optional[Path]) -> np.ndarray:
            if path is None:
                return np.zeros(cfg.latent_length, dtype=np.float32)
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            return self._resample_feat_to_latent_T(arr, start_sample)

        aux_pitch = self._load_pitch_salience(feats["pitch"], start_sample)  # [n_bins, T_lat]
        aux_rhythm = self._load_rhythm_aux(                                   # [n_rhythm_channels, T_lat]
            feats["rhythm"], feats["rhythm_multi"], start_sample
        )
        feat_pitch = self._load_pitch_salience_feature_T(feats["pitch"], start_sample)
        # feat_rhythm is the feature-rate rhythm tensor consumed by variants
        # with ``input_mode="feature"`` (e.g. Variant A). Prefer the 8-ch
        # multi-channel file (``rhythm_multi_feat_name``, produced by
        # ``03b_extract_rhythm_multi.py``); fall back to the 1-ch legacy
        # onset map (``rhythm_feat_name``) by broadcasting to the expected
        # channel count so downstream shape contracts hold.
        n_rhythm = int(cfg.n_rhythm_channels)
        if feats["rhythm_multi"] is not None:
            arr = np.load(feats["rhythm_multi"]).astype(np.float32)
            if arr.ndim != 2:
                raise RuntimeError(
                    f"{feats['rhythm_multi']}: expected [C, T_feat], got {arr.shape}"
                )
            if arr.shape[0] != n_rhythm:
                raise RuntimeError(
                    f"{feats['rhythm_multi']}: channel mismatch — file has "
                    f"{arr.shape[0]}, expected {n_rhythm}."
                )
            feat_rhythm = self._crop_feature_2d(
                arr, start_sample,
                expected_channels=n_rhythm, name="feat_rhythm",
            )
        elif feats["rhythm"] is not None:
            rhythm_raw = np.load(feats["rhythm"]).astype(np.float32).reshape(-1)
            one_ch = self._crop_feature_1d(rhythm_raw, start_sample)[None, :]   # [1, T_feat]
            feat_rhythm = np.tile(one_ch, (n_rhythm, 1)) if n_rhythm > 1 else one_ch
        else:
            feat_rhythm = np.zeros((n_rhythm, cfg.feat_length), dtype=np.float32)
        feat_mel = self._load_mel_feature_T(feats["spec"], start_sample)

        if cfg.prefer_envelope_timbre:
            aux_timbre = self._load_envelope_aux(feats["envelope"], start_sample)
        else:
            aux_timbre = self._load_timbre_aux(feats["timbre"], start_sample)
        feat_start = start_sample // cfg.feat_hop

        return {
            "audio": torch.from_numpy(audio_crop).unsqueeze(0),              # [1, N]
            "aux_pitch": torch.from_numpy(aux_pitch),                        # [n_bins, T_lat]
            "aux_rhythm": torch.from_numpy(aux_rhythm),                      # [n_rhythm_channels, T_lat]
            "aux_timbre": torch.from_numpy(aux_timbre),                      # [n_mfcc, T_lat]
            "feat_pitch": torch.from_numpy(feat_pitch),                      # [n_bins, T_feat]
            "feat_rhythm": torch.from_numpy(feat_rhythm),                    # [n_rhythm_channels, T_feat]
            "feat_mel": torch.from_numpy(feat_mel),                          # [n_mels, T_feat]
            "spec_path": str(feats["spec"]) if feats["spec"] is not None else "",
            "mel_linear_path": str(feats["mel_linear"]) if feats["mel_linear"] is not None else "",
            "feat_start": int(feat_start),
            "feat_length": int(cfg.feat_length),
            "track_id": str(d),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def stage1_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out = {
        "audio": torch.stack([b["audio"] for b in batch], dim=0),          # [B, 1, N]
        "aux_pitch": torch.stack([b["aux_pitch"] for b in batch], dim=0),  # [B, n_pitch_bins, T_lat]
        "aux_rhythm": torch.stack([b["aux_rhythm"] for b in batch], dim=0),
        "aux_timbre": torch.stack([b["aux_timbre"] for b in batch], dim=0), # [B, n_mfcc, T_lat]
        "feat_pitch": torch.stack([b["feat_pitch"] for b in batch], dim=0), # [B, n_pitch_bins, T_feat]
        "feat_rhythm": torch.stack([b["feat_rhythm"] for b in batch], dim=0),
        "feat_mel": torch.stack([b["feat_mel"] for b in batch], dim=0),     # [B, n_mels, T_feat]
        "spec_path": [b["spec_path"] for b in batch],
        "mel_linear_path": [b["mel_linear_path"] for b in batch],
        "feat_start": torch.tensor([int(b["feat_start"]) for b in batch], dtype=torch.long),
        "feat_length": torch.tensor([int(b["feat_length"]) for b in batch], dtype=torch.long),
        "track_id": [b["track_id"] for b in batch],
    }
    return out
