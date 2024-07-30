import os

import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from control_point_e import ControlPointE
from control_shapenet import ControlShapeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ControlPointE.load_from_checkpoint(
    "/scratch/noam/cntrl_pointe/07_28_2024_20_50_29_chair_train_chair_val_cd_filter_10th_switch_prob_0.5/epoch=399-step=38000.ckpt",
        lr=7e-5 * 0.4,
        dev=device,
        timesteps=1024,
        num_points=1024,
        batch_size=6,
        switch_prob=0.5,
        val_data_loader=None,
        cond_drop_prob=0.5,
    )
df = pd.read_csv("data.csv")
random_subset = df.sample(n=500, random_state=42)
chosen_indices = random_subset.index.to_numpy()
np.save('chosen_indices.npy', chosen_indices)
dataset = ControlShapeNet(
        df=random_subset,
        device=device,
        num_points=1024,
        batch_size=6,
    )
data_loader = DataLoader(
        dataset=dataset, batch_size=6, shuffle=False
    )
output = None
for batch in tqdm.tqdm(data_loader):
    prompts, source_latents = (batch["prompts"], batch["source_latents"].to(device))
    indices = torch.randperm(3072)[:2048]
    curr_output = model._sample(source_latents, prompts, 6)[:, :3, indices].detach().cpu()   
    if output is None:
        output = curr_output
    else:
        output = torch.cat((output, curr_output), dim=0)
torch.save(output, "output.pt")