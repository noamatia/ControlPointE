import tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset

from utils import *

PROMPTS = "prompts"
UTTERANCE = "utterance"
SOURCE_LATENTS = "source_latents"
TARGET_LATENTS = "target_latents"


def load_pc(shapenet_uid, num_points):
    src_dir = os.path.join(PCS_DIR, shapenet_uid + ".npz")
    pc = PointCloud.load_shapenet(src_dir)
    return pc.random_sample(num_points)


class ControlShapeNet(Dataset):
    def __init__(
        self,
        num_points: int,
        batch_size: int,
        df: pd.DataFrame,
        device: torch.device,
    ):
        super().__init__()
        self.prompts = []
        self.source_latents = []
        self.target_latents = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating data"):
            self._append_sample(row, num_points, device)
        self.set_length(batch_size)

    def _append_sample(self, row, num_points, device):
        prompt, source_uid, target_uid = (
            row[UTTERANCE],
            row[SOURCE_UID],
            row[TARGET_UID],
        )
        self.prompts.append(prompt)
        source_pc = load_pc(source_uid, num_points)
        target_pc = load_pc(target_uid, num_points)
        self.source_latents.append(source_pc.encode().to(device))
        self.target_latents.append(target_pc.encode().to(device))

    def set_length(self, batch_size, length=None):
        if length is None:
            self.length = len(self.prompts)
        else:
            assert length <= len(self.prompts)
            self.length = length
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
