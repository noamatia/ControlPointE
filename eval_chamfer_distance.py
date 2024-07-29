import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
tqdm.pandas()
from chamferdist import ChamferDistance


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chamfer_distance = ChamferDistance().to(device)

def eval_chamfer_distance(uid1, uid2, num_points=4096):
    path1 = f'/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering/{uid1}.npz'
    path2 = f'/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering/{uid2}.npz'
    data1 = np.load(path1)
    data2 = np.load(path2)
    pc1 = torch.tensor(data1['pointcloud']).to(device)
    pc2 = torch.tensor(data2['pointcloud']).to(device)
    pc1 = pc1[torch.randperm(pc1.size(0))[:num_points]]
    pc2 = pc2[torch.randperm(pc2.size(0))[:num_points]]
    data1 = pc1.unsqueeze(0)
    data2 = pc2.unsqueeze(0)
    return chamfer_distance(data1,data2).item()

df = pd.read_csv("/home/noamatia/repos/control_point_e/data/chair/train.csv")
df["chamfer_distance"] = df.progress_apply(lambda x: eval_chamfer_distance(x["source_uid"], x["target_uid"]), axis=1)
df.to_csv("/home/noamatia/repos/control_point_e/data/chair/train.csv", index=False)