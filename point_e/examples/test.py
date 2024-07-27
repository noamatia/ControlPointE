import sys

sys.path.insert(0, "/home/noamatia/repos/control_point_e/")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_name = "base40M-textvec"
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
base_model.load_state_dict(load_checkpoint(base_name, device))
base_model.load_state_dict(torch.load(f"/scratch/noam/pointe/model_weights.pt"))
base_model.eval()

sampler = PointCloudSampler(
    device=device,
    s_churn=[3],
    sigma_max=[120],
    num_points=[1024],
    sigma_min=[1e-3],
    models=[base_model],
    use_karras=[True],
    karras_steps=[64],
    guidance_scale=[3.0],
    diffusions=[base_diffusion],
    aux_channels=["R", "G", "B"],
    model_kwargs_key_filter=["texts"],
)


def build_experiment_dir(prompt1, prompt2, t, i):
    d = f"experiment2/p1_{prompt1}_p2_{prompt2}_t_{t}_i_{i}"
    os.makedirs(d, exist_ok=True)
    os.makedirs(d.replace("experiment2", "experiment2_objs"), exist_ok=True)
    return d


experimental1_t = 30
sampler.experiment2_t = experimental1_t
prompt_pairs = [
    ("a_chair", "a_chair_with_armrests"),
    ("a_chair", "a_chair_with_wheels"),
    ("a_chair", "a_chair_with_rounded_backrest"),
    ("a_chair", "a_chair_long_legs"),
    ("a_chair", "a_chair_with_thich_seat"),
    ("a_chair", "a_chair_with_spindles"),
    ("a_straight_chair", "a_straight_chair_with_a_pillow"),
    ("a_chair", "a_chair_with_two_legs"),
]
indices = [500, 750, 900]
percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

html = "<table>\n"
for prompt1, prompt2 in prompt_pairs:
    for i in tqdm(range(25), total=25):
        html += "<tr><td style='font-size: 24px;'>prompt</td>"
        for j in indices:
            html += f"<td style='font-size: 24px;'>{j}</td>"
        for percentile in percentiles:
            pp = f"{percentile}".replace(".", "_")
            html += f"<td style='font-size: 24px;'>{pp}</td>"
        html += "<td style='font-size: 24px;'>1024</td></tr>\n"
        sampler.experiment2_indices = None
        sampler.precentile = None
        os.environ["EXPERIMENT2_DIR"] = build_experiment_dir(
            prompt1, prompt2, experimental1_t, i
        )
        samples = None
        for x in tqdm(
            sampler.sample_batch_progressive(
                batch_size=2,
                model_kwargs=dict(
                    texts=[prompt1.replace("_", " "), prompt2.replace("_", " ")]
                ),
            )
        ):
            samples = x
        for j in indices:
            # selected_indices_path = os.path.join(os.environ["EXPERIMENT2_DIR"], "selected_indices.txt")
            # if not os.path.exists(selected_indices_path):
            #     continue
            # experiment2_indices = np.loadtxt(selected_indices_path, dtype=int)
            experiment2_indices = list(range(j))
            # ply_path = os.path.join(os.environ["EXPERIMENT2_DIR"].replace("experiment2", "experiment2_objs"), "1.ply")
            # pc = PointCloud.from_ply(ply_path)
            # pc.set_color_by_indices(experiment2_indices)
            # fig = plot_point_cloud(pc)
            # path = os.path.join(os.getenv("EXPERIMENT2_DIR"), f"1_selected_{j}.png")
            # fig.savefig(path)
            # plt.close()
            sampler.experiment2_indices = experiment2_indices
            samples = None
            for x in tqdm(
                sampler.sample_batch_progressive(
                    batch_size=2,
                    model_kwargs=dict(
                        texts=[prompt1.replace("_", " "), prompt2.replace("_", " ")]
                    ),
                )
            ):
                samples = x
        reversed_sorted_indices = np.load(
            os.path.join(os.getenv("EXPERIMENT2_DIR"), "sorted_indices.npy")
        )
        for percentile in percentiles:
            sampler.precentile = f"{percentile}".replace(".", "_")
            experiment2_indices = reversed_sorted_indices[
                : int(len(reversed_sorted_indices) * percentile)
            ]
            sampler.experiment2_indices = experiment2_indices
            samples = None
            for x in tqdm(
                sampler.sample_batch_progressive(
                    batch_size=2,
                    model_kwargs=dict(
                        texts=[prompt1.replace("_", " "), prompt2.replace("_", " ")]
                    ),
                )
            ):
                samples = x
        html += f"<tr><td>{prompt1}</td>"
        for j in indices:
            html += f"<td><img src=\"{os.path.join(os.getenv('EXPERIMENT2_DIR'), f'0_{j}.png')}\"></td>"
        for percentile in percentiles:
            pp = f"{percentile}".replace(".", "_")
            html += f"<td><img src=\"{os.path.join(os.getenv('EXPERIMENT2_DIR'), f'0_{pp}.png')}\"></td>"
        html += f"<td><img src=\"{os.path.join(os.getenv('EXPERIMENT2_DIR'), f'0.png')}\"></td></tr>\n"
        html += f"<tr><td>{prompt2}</td>"
        for j in indices:
            html += f"<td><img src=\"{os.path.join(os.getenv('EXPERIMENT2_DIR'), f'1_{j}.png')}\"></td>"
        for percentile in percentiles:
            pp = f"{percentile}".replace(".", "_")
            html += f"<td><img src=\"{os.path.join(os.getenv('EXPERIMENT2_DIR'), f'1_{pp}.png')}\"></td>"
        html += f"<td><img src=\"{os.path.join(os.getenv('EXPERIMENT2_DIR'), f'1.png')}\"></td></tr>\n"
        # html += f"<tr><td>{prompt2} selected</td>"
        # for j in indices:
        #     html += f"<td><img src=\"{os.path.join(os.getenv('EXPERIMENT2_DIR'), f'1_selected_{j}.png')}\"></td>"
        # html += f"<td></td></tr>\n"
html += "</table>"
with open("index.html", "w") as f:
    f.write(html)
