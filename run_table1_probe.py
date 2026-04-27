import os
import torch
import numpy as np
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

ROOT_DIR = "/root/autodl-tmp/SHAYI"
PART1_DIR = os.path.join(ROOT_DIR, "part_1")

def flatten_time_and_normalize(tensor, apply_norm=True):
    """
    把特征展平成 [N_samples, Features] 用于 sklearn 训练
    并使用 Instance Normalization 抹平全局能量带来的伪相关性
    """
    if tensor.ndim == 4: # Pitch [B, K, Bins, T]
        B, K, Bins, T = tensor.shape
        tensor = tensor.view(B, K*Bins, T)
        
    if apply_norm and tensor.ndim >= 3:
        # 对每一个时间步，在特征维度上做标准化 (Z-score)
        # 强迫探针只关注结构分布，不关注整体能量(响度)大小
        mu = tensor.mean(dim=1, keepdim=True)
        sigma = tensor.std(dim=1, keepdim=True) + 1e-8
        tensor = (tensor - mu) / sigma

    # [B, D, T] -> [B*T, D]
    if tensor.ndim == 3:
        return tensor.transpose(1, 2).reshape(-1, tensor.shape[1]).numpy()
    elif tensor.ndim == 2: # Timbre [B, D]
        return tensor.numpy()
    return tensor.numpy()

def main():
    print("1. 直接读取 100 个 Latents 及原始特征...")
    latent_dir = os.path.join(PART1_DIR, "data/latents_100")
    files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pt')])

    Z_p, Z_r, Z_t = [], [], []
    Phi_p, Phi_r, Phi_t = [], [], []

    for f in tqdm(files, desc="Loading Data"):
        data = torch.load(os.path.join(latent_dir, f), map_location="cpu")
        
        phi_p, phi_r, phi_t = data["phi_p"].unsqueeze(0), data["phi_r"].unsqueeze(0), data["phi_t"].unsqueeze(0)
        z_p, z_r, z_t = data["z_p"].unsqueeze(0), data["z_r"].unsqueeze(0), data["z_t"].unsqueeze(0)
        
        T = phi_p.shape[-1]
        # 音色特征在时间轴上扩展对齐
        z_t_expanded = z_t.unsqueeze(-1).expand(-1, -1, T)

        # 压平并执行去相关性归一化
        Z_p.append(flatten_time_and_normalize(z_p, apply_norm=True))
        Z_r.append(flatten_time_and_normalize(z_r, apply_norm=True))
        Z_t.append(flatten_time_and_normalize(z_t_expanded, apply_norm=True))
        
        Phi_p.append(flatten_time_and_normalize(phi_p, apply_norm=True))
        Phi_r.append(flatten_time_and_normalize(phi_r, apply_norm=True))
        Phi_t.append(flatten_time_and_normalize(phi_t, apply_norm=True))

    Z = {
        "zp": np.concatenate(Z_p), 
        "zr": np.concatenate(Z_r), 
        "zt": np.concatenate(Z_t)
    }
    Phi = {
        "p": np.concatenate(Phi_p), 
        "r": np.concatenate(Phi_r), 
        "t": np.concatenate(Phi_t)
    }

    print("2. 启动 RidgeCV 探针评估 (抽样 20000 帧)...")
    results = {z: {p: 0.0 for p in Phi} for z in Z}
    
    for z_name, z_data in Z.items():
        for phi_name, phi_data in Phi.items():
            # 为保证速度和显存不爆，随机抽样 20000 帧做回归
            n_samples = len(z_data)
            sample_size = min(20000, n_samples)
            idx = np.random.choice(n_samples, sample_size, replace=False)
            
            model = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(z_data[idx], phi_data[idx])
            score = max(0.0, model.score(z_data[idx], phi_data[idx])) # 裁剪掉负数 R2
            results[z_name][phi_name] = score

    print("\n" + "="*60)
    print("📋 请将以下代码直接复制进 LaTeX 的 Table 1 中：")
    print("="*60)
    print(f"    $z_p$ & \\textbf{{{results['zp']['p']:.3f}}} & {results['zp']['r']:.3f} & {results['zp']['t']:.3f} \\\\")
    print(f"    $z_r$ & {results['zr']['p']:.3f} & \\textbf{{{results['zr']['r']:.3f}}} & {results['zr']['t']:.3f} \\\\")
    print(f"    $z_t$ & {results['zt']['p']:.3f} & {results['zt']['r']:.3f} & \\textbf{{{results['zt']['t']:.3f}}} \\\\")
    print("="*60)

if __name__ == "__main__":
    main()