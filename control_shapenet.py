import copy
import tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset

from utils import *

SCALES = "scales"
PROMPTS = "prompts"
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
        self.batch_size = batch_size
        self.two_batch_size = 2 * batch_size
        self.data = {}
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating data"):
            self._append_sample(row, num_points, device)
        if len(self) % self.two_batch_size != 0:
            for i in range(len(self) % self.two_batch_size):
                self._append_sample(df.iloc[i % len(df)], num_points, device)

    def _append_sample(self, row, num_points, device):
        negative_prompt = row["negative_object"] + " without armrests"
        positive_prompt = row["positive_object"] + " with armrests"
        negative_latent = load_pc(row["negative_uid"], num_points).encode().to(device)
        positive_latent = load_pc(row["positive_uid"], num_points).encode().to(device)
        for scale, prompt, source_latent, target_latent in [
            (-1, negative_prompt, positive_latent, negative_latent),
            (1, positive_prompt, negative_latent, positive_latent),
        ]:
            index = self._eval_index(len(self))
            self.data[index] = {
                SCALES: scale,
                PROMPTS: prompt,
                SOURCE_LATENTS: source_latent,
                TARGET_LATENTS: target_latent,
            }

    def _eval_index(self, index):
        return (
            (index // self.two_batch_size) * self.two_batch_size
            + (index % 2) * self.batch_size
            + (index // 2) % self.batch_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def copy(self, length=None):
        new_dataset = copy.deepcopy(self)
        if length is not None:
            if length % self.two_batch_size != 0:
                length += self.two_batch_size - (length % self.two_batch_size)
            new_dataset.data = {
                self._eval_index(i): self.data[self._eval_index(i)]
                for i in range(length)
            }
        return new_dataset
