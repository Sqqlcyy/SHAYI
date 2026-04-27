import torch
import glob
import os
from torch.utils.data import Dataset
from shayi.training.train_dit import train_dit

torch.set_default_dtype(torch.float32)

class LatentDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {data_dir}!")
        print(f"Found {len(self.files)} files for training.")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        
        cond_len = 16
        cond = torch.zeros(cond_len, 768, dtype=torch.bfloat16) 
        cond_mask = torch.ones(cond_len, dtype=torch.bool)
        global_cond = torch.zeros(512, dtype=torch.bfloat16)
        
        return {
            "z_p": data["z_p"].to(torch.bfloat16),
            "z_r": data["z_r"].to(torch.bfloat16),
            "z_t": data["z_t"].to(torch.bfloat16),
            "cond_tokens": cond,
            "cond_mask": cond_mask,
            "global_cond": global_cond
        }

if __name__ == "__main__":
    dataset = LatentDataset("/root/autodl-tmp/SHAYI/part_1/data/latents")
    
    train_dit(
        dataset=dataset, 
        out_dir="./runs/dit_checkpoints_bf16", 
        batch_size=2,  # 🔥 核心修改：从 4 降到 2
        max_steps=40000
    )
