import torch
import glob

# 找到所有 checkpoint 并按 step 倒序排列 (比如 40000, 38000, 36000...)
ckpts = glob.glob("./runs/dit_checkpoints/shayi_dit_step*.pt")
ckpts.sort(key=lambda x: int(x.split('step')[-1].split('.pt')[0]), reverse=True)

print("🔍 开始诊断 Checkpoints...")

golden_ckpt = None
for ckpt_path in ckpts:
    print(f"检查: {ckpt_path} ...", end=" ")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']
    
    is_nan = False
    for name, param in state_dict.items():
        if torch.isnan(param).any() or torch.isinf(param).any():
            is_nan = True
            break
            
    if is_nan:
        print("❌ 发现 NaN (已损坏)")
    else:
        print("✅ 健康！(无 NaN/Inf)")
        golden_ckpt = ckpt_path
        break

if golden_ckpt:
    print("\n🎉 结论: 你的最强且健康的模型是 ->", golden_ckpt)
else:
    print("\n💀 结论: 全炸了 (大概率不可能)")
