import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class DenoisingDataset(Dataset):
    """RGB 去噪任务数据集，输入为退化图，目标为干净图。"""

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.sample_dirs = sorted(
            [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]
        )

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.sample_dirs[idx]
        degraded_path = os.path.join(sample_path, "degraded.npy")
        clean_path = os.path.join(sample_path, "clean.npy")
        mask_path = os.path.join(sample_path, "mask.npy")

        degraded_data = np.load(degraded_path).astype(np.float32)
        clean_data = np.load(clean_path).astype(np.float32)
        mask_data = np.load(mask_path).astype(np.float32)

        degraded_tensor = torch.from_numpy(degraded_data).permute(2, 0, 1).contiguous()
        clean_tensor = torch.from_numpy(clean_data).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask_data).unsqueeze(0)

        return degraded_tensor, clean_tensor, mask_tensor
