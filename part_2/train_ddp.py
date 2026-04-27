import torch
import glob
from torch.utils.data import Dataset
from shayi.training.train_dit import train_dit

# 1. 简单的 Dataset 读特征
class LatentDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        
        # 这里的 cond 必须和你真实提取的文本/音频条件对齐
        # 战时如果没弄好 cond，先用一个 0 向量代替！(只为了让它跑通)
        cond_len = 16
        cond = torch.zeros(cond_len, 768) 
        cond_mask = torch.ones(cond_len, dtype=torch.bool)
        global_cond = torch.zeros(512)
        
        return {
            "z_p": data["z_p"],
            "z_r": data["z_r"],
            "z_t": data["z_t"],
            "cond_tokens": cond,
            "cond_mask": cond_mask,
            "global_cond": global_cond
        }

if __name__ == "__main__":
    dataset = LatentDataset("./data/latents")
    
    # 开始训练！这里设置 BS=4，4张卡就是 16
    train_dit(
        dataset=dataset, 
        out_dir="./runs/dit_checkpoints", 
        batch_size=4, 
        max_steps=40000
    )