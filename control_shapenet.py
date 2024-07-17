import os
import tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset
from point_e.util.point_cloud import PointCloud

PROMPTS = "prompt"
SOURCE_UID = "source_uid"
TARGET_UID = "target_uid"
SOURCE_LATENTS = "source_latents"
TARGET_LATENTS = "target_latents"
PCS_DIR = "/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering"


def load_pc(num_points, uid):
    pc = PointCloud.load_shapenet(os.path.join(PCS_DIR, uid + ".npz"))
    return pc.random_sample(num_points)


class ControlShapeNet(Dataset):
    def __init__(
        self,
        num_points: int,
        batch_size: int,
        prompt_key: str,
        df: pd.DataFrame,
        device: torch.device,
    ):
        super().__init__()
        self.prompts = []
        self.source_latents = []
        self.target_latents = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating data"):
            self._append_sample(row, prompt_key, num_points, device)
        self._set_length(batch_size)

    def _append_sample(self, row, prompt_key, num_points, device):
        prompt, source_uid, target_uid = (
            row[prompt_key],
            row[SOURCE_UID],
            row[TARGET_UID],
        )
        self.prompts.append(prompt)
        source_pc = load_pc(num_points, source_uid)
        target_pc = load_pc(num_points, target_uid)
        self.source_latents.append(source_pc.encode().to(device))
        self.target_latents.append(target_pc.encode().to(device))

    def _set_length(self, batch_size):
        self.length = len(self.prompts)
        r = self.length % batch_size
        if r == 0:
            self.logical_length = self.length
        else:
            q = batch_size - r
            self.logical_length = self.length + q

    def __len__(self):
        return self.logical_length

    def __getitem__(self, logical_index):
        index = logical_index % self.length
        return {
            PROMPTS: self.prompts[index],
            SOURCE_LATENTS: self.source_latents[index],
            TARGET_LATENTS: self.target_latents[index],
        }
