import json
import tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset

from utils import *

PROMPTS = "prompts"
UTTERANCE = "utterance"
SOURCE_MASKS = "source_masks"
TARGET_MASKS = "target_masks"
SOURCE_LATENTS = "source_latents"
TARGET_LATENTS = "target_latents"


def partnet_metadata_path(uid):
    return os.path.join(PARTNET_METADATA_DIR, uid, PARTNET_METADATA_JSON)


def load_partnet_metadata(uid, part):
    path = partnet_metadata_path(uid)
    with open(path, "r") as f:
        data = json.load(f)
    masked_labels = data["masked_labels"][part]
    return masked_labels, data["partnet_uid"]


# def load_pc(partnet_uid, shapenet_uid, num_points, masked_labels):
#     src_dir = os.path.join(PARTNET_DIR, partnet_uid)
#     pc = load_partnet(src_dir, shapenet_uid, masked_labels)
#     return pc.random_sample(num_points)

def load_pc(shapenet_uid, num_points):
    pc = PointCloud.load_shapenet(os.path.join(PCS_DIR, shapenet_uid + ".npz"))
    return pc.random_sample(num_points)


class ControlShapeNet(Dataset):
    def __init__(
        self,
        # part: str,
        num_points: int,
        batch_size: int,
        df: pd.DataFrame,
        device: torch.device,
    ):
        super().__init__()
        self.prompts = []
        # self.source_masks = []
        # self.target_masks = []
        self.source_latents = []
        self.target_latents = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Creating data"):
            # self._append_sample(row, num_points, device, part)
            self._append_sample(row, num_points, device)
        self.set_length(batch_size)

    # def _append_sample(self, row, num_points, device, part):
    def _append_sample(self, row, num_points, device):
        prompt, source_uid, target_uid = (
            row[UTTERANCE],
            row[SOURCE_UID],
            row[TARGET_UID],
        )
        self.prompts.append(prompt)
        # source_masked_labels, source_partnet_uid = load_partnet_metadata(
        #     source_uid, part
        # )
        # target_masked_labels, target_partnet_uid = load_partnet_metadata(
        #     target_uid, part
        # )
        # source_pc = load_pc(
        #     source_partnet_uid,
        #     source_uid,
        #     num_points,
        #     source_masked_labels,
        # )
        # target_pc = load_pc(
        #     target_partnet_uid,
        #     target_uid,
        #     num_points,
        #     target_masked_labels,
        # )
        # self.source_masks.append(source_pc.encode_mask().to(device))
        # self.target_masks.append(target_pc.encode_mask().to(device))
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
            # SOURCE_MASKS: self.source_masks[index],
            # TARGET_MASKS: self.target_masks[index],
            SOURCE_LATENTS: self.source_latents[index],
            TARGET_LATENTS: self.target_latents[index],
        }
