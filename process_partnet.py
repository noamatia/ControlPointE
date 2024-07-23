import tqdm
import json
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from point_e.util.plotting import plot_point_cloud

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="chair_arm")
    parser.add_argument("--data", type=str, default="chair/train")
    args = parser.parse_args()
    return args


def find_leaf_ids(node, target_name):
    leaf_ids = []
    if "name" in node and node["name"] == target_name:
        for child in node.get("children", []):
            if "children" in child:
                leaf_ids.extend(find_leaf_ids(child, target_name))
            else:
                leaf_ids.append(child["id"])
    else:
        for child in node.get("children", []):
            leaf_ids.extend(find_leaf_ids(child, target_name))
    return leaf_ids


def main(args):
    df = pd.read_csv(os.path.join(DATA_DIR, args.data + ".csv"))
    model_id_to_shapenet_uid = {}
    for _, row in df.iterrows():
        model_id_to_shapenet_uid[row[SOURCE_UID].split("/")[-1]] = row[SOURCE_UID]
        model_id_to_shapenet_uid[row[TARGET_UID].split("/")[-1]] = row[TARGET_UID]

    partnet_uids = os.listdir(PARTNET_DIR)
    for partnet_uid in tqdm.tqdm(
        partnet_uids, total=len(partnet_uids), desc="Processing PartNet"
    ):
        src_dir = os.path.join(PARTNET_DIR, partnet_uid)
        with open(os.path.join(src_dir, "meta.json"), "r") as f:
            metadata = json.load(f)
        model_id = metadata["model_id"]
        if model_id not in model_id_to_shapenet_uid:
            continue

        with open(os.path.join(src_dir, "result.json"), "r") as f:
            metadata = json.load(f)
        masked_labels = []
        for item in metadata:
            masked_labels.extend(find_leaf_ids(item, args.part))

        tgt_dir = os.path.join(PARTNET_METADATA_DIR, model_id_to_shapenet_uid[model_id])
        os.makedirs(tgt_dir, exist_ok=True)

        json_path = os.path.join(tgt_dir, PARTNET_METADATA_JSON)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                json_data = json.load(f)
            if json_data["partnet_uid"] != partnet_uid:
                continue
        else:
            json_data = {"masked_labels": {}, "partnet_uid": partnet_uid}

        json_data["masked_labels"][args.part] = masked_labels
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        pc = load_partnet(src_dir, model_id_to_shapenet_uid[model_id], masked_labels)
        fig = plot_point_cloud(pc, theta=np.pi * 1 / 2)
        fig.savefig(os.path.join(tgt_dir, "pc.png"))
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
