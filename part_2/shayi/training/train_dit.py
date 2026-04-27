import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from shayi.models.hifra_dit import HiFRADiT, HiFRADiTConfig
from shayi.losses.flow_matching import flow_matching_step

def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if vals[0] is None:
            out[k] = None
        else:
            out[k] = torch.stack(vals, dim=0)
    return out

def train_dit(
    dataset,
    out_dir,
    batch_size=8,
    lr=1e-4,
    max_steps=30000,
    log_every=50,
    save_every=2000,
    global_cond_dim=512,
):
    # 1. 换成 fp16 兼容性最好，绝不会报错
    accelerator = Accelerator(mixed_precision="bf16") 

    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)

    cfg = HiFRADiTConfig(
        T=861,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        local_cond_dim=768,
        global_cond_dim=global_cond_dim
    )
    model = HiFRADiT(cfg)
    
    if accelerator.is_main_process:
        print(f"🔥 HiFRA-T Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model, opt, loader, sched = accelerator.prepare(model, opt, loader, sched)

    step = 0
    model.train()
    
    if accelerator.is_main_process:
        print("🚀 Starting DDP Training Loop...")
        
    while step < max_steps:
        for batch in loader:
            max_T = 861
            if batch["z_p"].shape[-1] > max_T:
                start_idx = torch.randint(0, batch["z_p"].shape[-1] - max_T + 1, (1,)).item()
                batch["z_p"] = batch["z_p"][..., start_idx : start_idx + max_T]
                batch["z_r"] = batch["z_r"][..., start_idx : start_idx + max_T]

            opt.zero_grad()
            
            # 🔥 核心修复：用 autocast 包裹前向和 Loss 计算
            with accelerator.autocast():
                loss, logs = flow_matching_step(model, batch)
                
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            opt.step()
            sched.step()

            step += 1
            
            if step % log_every == 0 and accelerator.is_main_process:
                print(f"[{step}/{max_steps}] " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()))

            if step % save_every == 0 and accelerator.is_main_process:
                ckpt_path = os.path.join(out_dir, f"shayi_dit_step{step}.pt")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    "step": step,
                    "cfg": cfg.__dict__,
                    "model": unwrapped_model.state_dict(),
                }, ckpt_path)
                print(f"✅ Saved Checkpoint: {ckpt_path}")

            if step >= max_steps:
                break

    if accelerator.is_main_process:
        ckpt_path = os.path.join(out_dir, "shayi_dit_final.pt")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            "step": step,
            "cfg": cfg.__dict__,
            "model": unwrapped_model.state_dict(),
        }, ckpt_path)
        print(f"🎉 Training Complete! Saved: {ckpt_path}")
