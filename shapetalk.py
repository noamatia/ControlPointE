import os
import tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset
from point_e.util.point_cloud import PointCloud

PROMPTS = "prompts"
UTTERANCE = "utterance"
SOURCE_UID = "source_uid"
TARGET_UID = "target_uid"
NEGATIVE_UID = "negative_uid"
POSITIVE_UID = "positive_uid"
SWITCH_PROMPTS = "switch_prompts"
SOURCE_LATENTS = "source_latents"
TARGET_LATENTS = "target_latents"
NEGATIVE_PROMPTS = "negative_prompts"
POSITIVE_PROMPTS = "positive_prompts"
NEGATIVE_LATENTS = "negative_latents"
POSITIVE_LATENTS = "positive_latents"
NEGATIVE_UTTERANCE = "negative_utterance"
POSITIVE_UTTERANCE = "positive_utterance"
LLAMA3_WNLEMMA_UTTERANCE = "llama3_wnlemma_utterance"
PCS_DIR = "/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering"


def load_pc(shapenet_uid, num_points):
    pc = PointCloud.load_shapenet(os.path.join(PCS_DIR, shapenet_uid + ".npz"))
    return pc.random_sample(num_points)


class ShapeTalk(Dataset):
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
        self.switch_prompts = []
        self.source_latents = []
        self.target_latents = []
        for _, row in tqdm.tqdm(
            df.iterrows(), total=len(df), desc="Creating ShapeTalk dataset"
        ):
            self._append_sample(row, prompt_key, num_points, device)
        self.set_length(batch_size)

    def _append_sample(self, row, prompt_key, num_points, device):
        source_uid, target_uid, prompt, random_wnlemma = (
            row[SOURCE_UID],
            row[TARGET_UID],
            row[prompt_key],
            row["random_wnlemma"],
        )
        self.prompts.append(prompt)
        if prompt_key == LLAMA3_WNLEMMA_UTTERANCE:
            self.switch_prompts.append(random_wnlemma)
        elif prompt_key == UTTERANCE:
            self.switch_prompts.append(prompt)
        else:
            raise ValueError(f"Unknown prompt_key: {prompt}")
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
            SWITCH_PROMPTS: self.switch_prompts[index],
        }


class ShapeTalkLoRA(Dataset):
    def __init__(
        self,
        num_points: int,
        batch_size: int,
        df: pd.DataFrame,
        device: torch.device,
    ):
        super().__init__()
        self.negative_prompts = []
        self.positive_prompts = []
        self.negative_latents = []
        self.positive_latents = []
        for _, row in tqdm.tqdm(
            df.iterrows(), total=len(df), desc="Creating ShapeTalkLoRA dataset"
        ):
            self._append_sample(row, num_points, device)
        self.set_length(batch_size)

    def _append_sample(self, row, num_points, device):
        negative_uid, positive_uid, negative_prompt, positive_prompt = (
            row[NEGATIVE_UID],
            row[POSITIVE_UID],
            row[NEGATIVE_UTTERANCE],
            row[POSITIVE_UTTERANCE],
        )
        self.negative_prompts.append(negative_prompt)
        self.positive_prompts.append(positive_prompt)
        negative_pc = load_pc(negative_uid, num_points)
        positive_pc = load_pc(positive_uid, num_points)
        self.negative_latents.append(negative_pc.encode().to(device))
        self.positive_latents.append(positive_pc.encode().to(device))

    def set_length(self, batch_size, length=None):
        if length is None:
            self.length = len(self.negative_prompts)
        else:
            assert length <= len(self.negative_prompts)
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
            NEGATIVE_PROMPTS: self.negative_prompts[index],
            POSITIVE_PROMPTS: self.positive_prompts[index],
            NEGATIVE_LATENTS: self.negative_latents[index],
            POSITIVE_LATENTS: self.positive_latents[index],
        }
