import os
import sys
import torch
import torchaudio
from tqdm import tqdm
import yaml

# =====================================================================
# 1. 强行打通 part_1 和 part_2 的环境变量 (无需再管乱七八糟的引用)
# =====================================================================
ROOT_DIR = "/root/autodl-tmp/SHAYI"
PART1_DIR = os.path.join(ROOT_DIR, "part_1")
PART2_DIR = os.path.join(ROOT_DIR, "part_2")
sys.path.insert(0, PART1_DIR)
sys.path.insert(0, PART2_DIR)

from shayi.models.hifra_dit import HiFRADiT, HiFRADiTConfig
from runtime import build_model  # part_1 里你自己的神仙函数

# =====================================================================
# 2. FactorLock 推断核心逻辑 (带能量旋钮)
# =====================================================================
@torch.no_grad()
def factorlock_with_knobs(
    model, z_orig_p, z_orig_t, z_orig_r, 
    edit_target="rhythm", knob_scale=1.0, steps=25, device="cuda"
):
    model.eval()
    B = z_orig_p.shape[0]
    
    # 初始纯高斯噪声
    z_tau_p = torch.randn_like(z_orig_p)
    z_tau_r = torch.randn_like(z_orig_r)
    z_tau_t = torch.randn_like(z_orig_t)

    # Dummy Conditions (因为我们没训 T2A，这是“无条件纯几何编辑”)
    cond_tokens = torch.zeros(B, 16, 768, device=device, dtype=torch.bfloat16)
    global_cond = torch.zeros(B, 512, device=device, dtype=torch.bfloat16)

    dt = 1.0 / steps
    print(f"🎛️ 开始 FactorLock 混音... [修改: {edit_target} | 强度旋钮: {knob_scale}x]")
    
    for i in tqdm(range(steps), desc="ODE Solving"):
        tau = torch.tensor([i * dt] * B, device=device, dtype=torch.bfloat16)
        
        # 🔒 FactorLock 锚定
        tau_p_view, tau_r_view, tau_t_view = tau.view(-1,1,1,1), tau.view(-1,1,1), tau.view(-1,1)
        if edit_target != "pitch":
            z_tau_p = (1 - tau_p_view) * torch.randn_like(z_orig_p) + tau_p_view * z_orig_p
        if edit_target != "rhythm":
            z_tau_r = (1 - tau_r_view) * torch.randn_like(z_orig_r) + tau_r_view * z_orig_r
        if edit_target != "timbre":
            z_tau_t = (1 - tau_t_view) * torch.randn_like(z_orig_t) + tau_t_view * z_orig_t

        # 🏃 预测速度 (🔥 加上 autocast，彻底杜绝数据类型冲突！)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            v_p, v_r, v_t = model(z_tau_p, z_tau_r, z_tau_t, tau, cond_tokens=cond_tokens, global_cond=global_cond)
        
        # 🎚️ Energy Guidance 能量旋钮放大
        if edit_target == "pitch":
            v_p = v_p + (knob_scale - 1.0) * v_p.mean(dim=(1, 2, 3), keepdim=True)
        elif edit_target == "rhythm":
            v_r = v_r + (knob_scale - 1.0) * v_r.mean(dim=(1, 2), keepdim=True)
        elif edit_target == "timbre":
            v_t = v_t + (knob_scale - 1.0) * v_t.mean(dim=-1, keepdim=True)

        # 🎯 Euler 更新
        if edit_target == "pitch": z_tau_p += v_p * dt
        elif edit_target == "rhythm": z_tau_r += v_r * dt
        elif edit_target == "timbre": z_tau_t += v_t * dt

    return z_tau_p, z_tau_r, z_tau_t


# =====================================================================
# 3. 组装流水线并输出声音
# =====================================================================
def main():
    device = "cuda"
    os.makedirs("./demo_workspace", exist_ok=True)

    print("=> 1. 加载 40000 步的黄金 DiT...")
    dit_cfg = HiFRADiTConfig(T=861, hidden_dim=768, depth=12, num_heads=12, local_cond_dim=768, global_cond_dim=512)
    dit_model = HiFRADiT(dit_cfg).to(device).to(torch.bfloat16)
    dit_ckpt = torch.load(os.path.join(PART2_DIR, "runs/dit_checkpoints_bf16/shayi_dit_final.pt"), map_location="cpu")
    dit_model.load_state_dict(dit_ckpt["model"])
    dit_model.eval()

    print("=> 2. 加载 Analytic AE...")
    config_path = os.path.join(PART1_DIR, "configs/train_a_nocontent_full.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    
    ae_model = build_model(raw_cfg["model"])
    ae_ckpt = torch.load(os.path.join(PART1_DIR, "runs/stage1_nocontent/ckpt_step00040000.pt"), map_location="cpu")
    
    # 兼容两种保存格式
    state_dict = ae_ckpt["model"] if "model" in ae_ckpt else ae_ckpt
    ae_model.load_state_dict(state_dict, strict=False)
    ae_model.to(device).eval()

    print("=> 3. 寻找真实原曲特征...")
    latent_files = [f for f in os.listdir(os.path.join(PART1_DIR, "data/latents")) if f.endswith('.pt')]
    if not latent_files:
        raise RuntimeError("找不到 latent 文件！请检查 part_1/data/latents 目录。")
    
    # 取第一个文件做 Demo
    test_file = os.path.join(PART1_DIR, "data/latents", latent_files[0])
    print(f"   选中锚点文件: {latent_files[0]}")
    real_latent = torch.load(test_file, map_location=device)
    
    z_orig_p = real_latent["z_p"].unsqueeze(0).to(torch.bfloat16)
    z_orig_r = real_latent["z_r"].unsqueeze(0).to(torch.bfloat16)
    z_orig_t = real_latent["z_t"].unsqueeze(0).to(torch.bfloat16)

    # ---------------- 🧪 核心生成实验 ----------------
    # 实验：FactorLock 锁住音高+音色，重新生成节奏，并且强度拉到 2 倍！
    gen_p, gen_r, gen_t = factorlock_with_knobs(
        dit_model, z_orig_p, z_orig_t, z_orig_r, 
        edit_target="rhythm", knob_scale=2.0, steps=25, device=device
    )

    print("=> 4. 交付 AE 渲染出声...")

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            
            # --- 1. 渲染原版音频 (Baseline) ---
            print("   正在渲染原曲锚点 (AE Reconstruction)...")
            p_orig_f32 = z_orig_p.to(torch.float32)
            r_orig_f32 = z_orig_r.to(torch.float32)
            t_orig_f32 = z_orig_t.to(torch.float32)
            wav_orig = ae_model.decode(p_orig_f32, r_orig_f32, t_orig_f32)
            
            # --- 2. 渲染 FactorLock 编辑后的音频 ---
            print("   正在渲染编辑后的音乐 (FactorLock 2x)...")
            p_gen_f32 = gen_p.to(torch.float32)
            r_gen_f32 = gen_r.to(torch.float32)
            t_gen_f32 = gen_t.to(torch.float32)
            wav_edit = ae_model.decode(p_gen_f32, r_gen_f32, t_gen_f32)

    # ================= 处理并保存 =================
    def save_wav(wav_tensor, suffix):
        if isinstance(wav_tensor, dict): wav_tensor = wav_tensor['audio']
        elif isinstance(wav_tensor, tuple): wav_tensor = wav_tensor[0]
        
        wav_tensor = wav_tensor.cpu().squeeze(0) # 变成 [1, Samples]
        if wav_tensor.ndim == 1: wav_tensor = wav_tensor.unsqueeze(0)
        
        out_name = latent_files[0].replace('.pt', f'_{suffix}.wav')
        out_path = os.path.join("./demo_workspace", out_name)
        torchaudio.save(out_path, wav_tensor, 32000)
        print(f"   💾 保存成功: {out_path}")

    save_wav(wav_orig, "ORIGINAL")
    save_wav(wav_edit, "EDITED_rhythm_2x")
    print("✅ 全部搞定！快去 demo_workspace 下载这两个文件对比听听看！")

if __name__ == "__main__":
    main()
