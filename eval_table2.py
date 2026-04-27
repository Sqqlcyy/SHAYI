import os
import glob
import librosa
import numpy as np
from tqdm import tqdm

# 你保存 70 个 wav 文件的目录
WAV_DIR = "/root/autodl-tmp/SHAYI/infer"

def extract_features(wav_path):
    """提取纯粹的客观物理特征"""
    y, sr = librosa.load(wav_path, sr=32000)
    
    # 1. Pitch: Chroma CQT (反映和弦与音高分布)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    
    # 2. Rhythm: Onset Strength (反映鼓点和瞬态节奏)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    
    # 3. Timbre: MFCC 排除第一维能量 (反映音色和频谱包络)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)[1:]
    
    return chroma.T, onset.reshape(-1, 1), mfcc.T

def relative_shift(feat_edit, feat_orig):
    """计算相对变化率 (%)"""
    # 对齐时间轴长度（以防万一有微小截断）
    min_len = min(feat_edit.shape[0], feat_orig.shape[0])
    fe, fo = feat_edit[:min_len], feat_orig[:min_len]
    
    diff_norm = np.linalg.norm(fe - fo)
    orig_norm = np.linalg.norm(fo) + 1e-8
    return (diff_norm / orig_norm) * 100

def main():
    # 找到所有的 original baseline 文件
    orig_files = sorted(glob.glob(os.path.join(WAV_DIR, "*_ORIGINAL.wav")))
    
    conditions = ["RHYTHM_1.5", "RHYTHM_2.0", "PITCH_1.8", "TIMBRE_1.8"]
    results = {c: {"target": [], "off_p": [], "off_r": [], "off_t": []} for c in conditions}

    print("🎧 开始对 WAV 波形进行硬核物理特征审查...")
    
    for orig_path in tqdm(orig_files):
        base_name = os.path.basename(orig_path).replace("_ORIGINAL.wav", "")
        
        # 提取原曲客观特征
        orig_p, orig_r, orig_t = extract_features(orig_path)
        
        for cond in conditions:
            edit_path = os.path.join(WAV_DIR, f"{base_name}_{cond}.wav")
            if not os.path.exists(edit_path):
                continue
                
            # 提取编辑后音频的客观特征
            edit_p, edit_r, edit_t = extract_features(edit_path)
            
            p_shift = relative_shift(edit_p, orig_p)
            r_shift = relative_shift(edit_r, orig_r)
            t_shift = relative_shift(edit_t, orig_t)
            
            # 分类记录
            if "PITCH" in cond:
                results[cond]["target"].append(p_shift)
                results[cond]["off_r"].append(r_shift)
                results[cond]["off_t"].append(t_shift)
            elif "RHYTHM" in cond:
                results[cond]["target"].append(r_shift)
                results[cond]["off_p"].append(p_shift)
                results[cond]["off_t"].append(t_shift)
            elif "TIMBRE" in cond:
                results[cond]["target"].append(t_shift)
                results[cond]["off_p"].append(p_shift)
                results[cond]["off_r"].append(r_shift)

    print("\n" + "="*80)
    print("📋 Table 2: FactorLock Waveform Conservation (True Physics Metrics)")
    print("="*80)
    def avg(lst): return sum(lst)/len(lst) if lst else 0.0

    print(f"    Baseline & 0.0\\% & 0.0\\% & 0.0\\% \\\\")
    
    r15 = results["RHYTHM_1.5"]
    print(f"    Rhythm ($\\gamma=1.5$) & \\textbf{{{avg(r15['target']):.1f}\\%}} & {avg(r15['off_p']):.1f}\\% & {avg(r15['off_t']):.1f}\\% \\\\")
    
    r20 = results["RHYTHM_2.0"]
    print(f"    Rhythm ($\\gamma=2.0$) & \\textbf{{{avg(r20['target']):.1f}\\%}} & {avg(r20['off_p']):.1f}\\% & {avg(r20['off_t']):.1f}\\% \\\\")
    
    p18 = results["PITCH_1.8"]
    print(f"    Pitch ($\\gamma=1.8$)  & \\textbf{{{avg(p18['target']):.1f}\\%}} & - & {avg(p18['off_t']):.1f}\\% \\\\")
    print("="*80)

if __name__ == "__main__":
    main()
