import os
import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm

# --- 路径黑魔法 ---
ROOT_DIR = "/root/autodl-tmp/SHAYI"
PART1_DIR = os.path.join(ROOT_DIR, "part_1")
PART2_DIR = os.path.join(ROOT_DIR, "part_2")
sys.path.insert(0, PART1_DIR)
sys.path.insert(0, PART2_DIR)

from dataset import Stage1DataConfig, Stage1AEDataset, stage1_collate
from runtime import build_model as build_ae_model
from part_1.stage2.ay_loss_v2 import AnalyticYieldingLossV2, AYConfigV2
from shayi.models.hifra_dit import HiFRADiT, HiFRADiTConfig
from shayi.losses.flow_matching import flow_matching_step

def build_dataloader(data_cfg: dict, batch_size: int):
    roots = [Path(os.path.join(PART1_DIR, r)) for r in data_cfg.get("roots", [])]
    track_list_path = os.path.join(PART1_DIR, data_cfg.get("train_list")) if data_cfg.get("train_list") else None
    
    cfg = Stage1DataConfig(
        roots=roots,
        track_list=Path(track_list_path) if track_list_path else None,
        sample_rate=data_cfg.get("sample_rate", 32000),
        crop_seconds=3.0,  
        feat_hop=data_cfg.get("feat_hop", 512), 
        latent_hop=data_cfg.get("latent_hop", 512),
        normalize_audio=data_cfg.get("normalize_audio", True), 
        augment=False,
        pitch_feat_name=data_cfg.get("pitch_feat_name", "pitch_salience_instru_nondrum.npy"),
        rhythm_feat_name=data_cfg.get("rhythm_feat_name", "rhythm_instru.npy"),
        rhythm_multi_feat_name=data_cfg.get("rhythm_multi_feat_name", "rhythm_multi_instru.npy"),
        timbre_feat_name=data_cfg.get("timbre_feat_name", "mfcc_instru.npy"),
        envelope_feat_name=data_cfg.get("envelope_feat_name", "envelope_instru.npy"),
        prefer_envelope_timbre=data_cfg.get("prefer_envelope_timbre", False),
        n_pitch_bins=data_cfg.get("n_pitch_bins", 588),
        n_mels=data_cfg.get("n_mels", 128),
        n_mfcc=data_cfg.get("n_mfcc", 20),
        n_rhythm_channels=data_cfg.get("n_rhythm_channels", 8),
        n_envelope_dim=data_cfg.get("n_envelope_dim", 80),
        allow_missing_aux=True
    )
    ds = Stage1AEDataset(cfg)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=stage1_collate, drop_last=True)


def main():
    # 🔥 关闭所有恶心的检查，回归最纯粹的 DDP (因为我们要手动剪枝)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False, 
        static_graph=False
    )
    
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    set_seed(42)

    if accelerator.is_main_process:
        print("🚀 [Stage 2] 初始化 Online AE + DiT 端到端训练...")
        os.makedirs("./runs/stage2_e2e_ddp", exist_ok=True)

    # 1. 载入 AE 
    with open(os.path.join(PART1_DIR, "configs/train_a_nocontent_full.yaml"), "r") as f:
        ae_raw_cfg = yaml.safe_load(f)
    
    ae_model = build_ae_model(ae_raw_cfg["model"])
    ae_ckpt = torch.load(os.path.join(PART1_DIR, "runs/stage1_nocontent/ckpt_step00040000.pt"), map_location="cpu")
    ae_model.load_state_dict(ae_ckpt["model"] if "model" in ae_ckpt else ae_ckpt, strict=False)
    ae_model.to(device)

    ayi_cfg = AYConfigV2(w_equiv=1.0, w_inv=1.0, w_comm=0.5)
    ayi_loss_fn = AnalyticYieldingLossV2(ayi_cfg).to(device)

    # 2. 载入 DiT
    dit_cfg = HiFRADiTConfig(T=861, hidden_dim=768, depth=12, num_heads=12, local_cond_dim=768, global_cond_dim=512)
    dit_model = HiFRADiT(dit_cfg).to(device)
    dit_ckpt = torch.load(os.path.join(PART2_DIR, "runs/dit_checkpoints_bf16/shayi_dit_final.pt"), map_location="cpu")
    dit_model.load_state_dict(dit_ckpt["model"])
    
    dit_model.blocks.gradient_checkpointing = True 

    # ================================================================
    # 🚨 神级黑科技：在交给 DDP 之前，模拟一次计算，手动切断闲置参数的梯度！
    # ================================================================
    if accelerator.is_main_process:
        print("🔍 正在执行静态计算图动态剪枝 (消灭 DDP 报错)...")
        
    dummy_loader = build_dataloader(ae_raw_cfg["data"], batch_size=1)
    dummy_batch = next(iter(dummy_loader))
    dummy_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in dummy_batch.items()}
    
    ae_model.train()
    dit_model.train()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        enc_anchor = ae_model(dummy_batch["feat_pitch"], dummy_batch["feat_rhythm"], dummy_batch["feat_mel"])
        ayi_out = ayi_loss_fn(ae_model, enc_anchor, dummy_batch["audio"], dummy_batch)
        loss_ayi = ayi_out["ay/total"]
        
        z_p, z_r, z_t = enc_anchor["z_p"], enc_anchor["z_r"], enc_anchor["z_t"]
        if z_p.shape[-1] > 512: z_p, z_r = z_p[..., :512], z_r[..., :512]
            
        dit_batch = {
            "z_p": z_p, "z_r": z_r, "z_t": z_t,
            "cond_tokens": torch.zeros(z_p.shape[0], 16, 768, device=device, dtype=torch.bfloat16),
            "global_cond": torch.zeros(z_p.shape[0], 512, device=device, dtype=torch.bfloat16)
        }
        loss_fm, _ = flow_matching_step(dit_model, dit_batch)
        loss_total = loss_fm + 0.5 * loss_ayi

    loss_total.backward()

    # 🔥 冻结所有没有产生梯度的参数 (比如 AE Decoder)
    for param in ae_model.parameters():
        if param.grad is None:
            param.requires_grad = False
    for param in dit_model.parameters():
        if param.grad is None:
            param.requires_grad = False

    # 清空垃圾梯度，假装无事发生
    ae_model.zero_grad()
    dit_model.zero_grad()
    
    if accelerator.is_main_process:
        print("✅ 闲置参数剪枝完毕！已完美规避 DDP 死锁。")
    # ================================================================

    # 3. 初始化优化器 (只包含那些 requires_grad=True 的健康参数！)
    combined_params = [p for p in ae_model.parameters() if p.requires_grad] + \
                      [p for p in dit_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(combined_params, lr=2e-5, weight_decay=0.01)

    # 用 BS=1 正式开跑
    loader = build_dataloader(ae_raw_cfg["data"], batch_size=1) 
    
    # 将清理干净的模型交给 DDP！
    ae_model, dit_model, opt, loader = accelerator.prepare(ae_model, dit_model, opt, loader)

    step = 0
    max_steps = 40000

    while step < max_steps:
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            opt.zero_grad()

            with accelerator.autocast():
                feat_pitch = batch["feat_pitch"]
                feat_rhythm = batch["feat_rhythm"]
                feat_mel = batch["feat_mel"]
                audio = batch["audio"]
                
                enc_anchor = ae_model(feat_pitch, feat_rhythm, feat_mel)
                
                unwrapped_ae = accelerator.unwrap_model(ae_model)
                ayi_out = ayi_loss_fn(unwrapped_ae, enc_anchor, audio, batch)
                loss_ayi = ayi_out["ay/total"]

                z_p, z_r, z_t = enc_anchor["z_p"], enc_anchor["z_r"], enc_anchor["z_t"]
                
                max_T = 512
                if z_p.shape[-1] > max_T:
                    z_p, z_r = z_p[..., :max_T], z_r[..., :max_T]
                
                dit_batch = {
                    "z_p": z_p, "z_r": z_r, "z_t": z_t,
                    "cond_tokens": torch.zeros(z_p.shape[0], 16, 768, device=device, dtype=torch.bfloat16),
                    "global_cond": torch.zeros(z_p.shape[0], 512, device=device, dtype=torch.bfloat16)
                }
                loss_fm, _ = flow_matching_step(dit_model, dit_batch)

                loss_total = loss_fm + 0.5 * loss_ayi

            accelerator.backward(loss_total)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(combined_params, 1.0)
            opt.step()
            step += 1

            if step % 10 == 0 and accelerator.is_main_process:
                print(f"🌍 [Stage 2 E2E] Step {step}/{max_steps} | Total: {loss_total.item():.4f} | FM(DiT): {loss_fm.item():.4f} | AYI(AE): {loss_ayi.item():.4f}")

            if step % 2000 == 0 and accelerator.is_main_process:
                torch.save({
                    "ae_model": accelerator.unwrap_model(ae_model).state_dict(),
                    "dit_model": accelerator.unwrap_model(dit_model).state_dict()
                }, f"./runs/stage2_e2e_ddp/e2e_step_{step}.pt")

            if step >= max_steps:
                break

if __name__ == "__main__":
    main()