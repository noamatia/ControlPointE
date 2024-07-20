import os
import numpy as np
import open3d as o3d
from tqdm.auto import tqdm

experimental1_t = 30
prompt1, prompt2 = "a_chair", "a_chair_with_armrests"


def build_experiment_dir(prompt1, prompt2, t, i):
    d = f"/home/noamatia/repos/control_point_e/point_e/examples/experiment2/p1_{prompt1}_p2_{prompt2}_t_{t}_i_{i}"
    os.makedirs(d, exist_ok=True)
    return d


for i in tqdm(range(25), total=25):
    os.environ["EXPERIMENT2_DIR"] = build_experiment_dir(
        prompt1, prompt2, experimental1_t, i
    )
    pc_path = os.path.join(os.environ["EXPERIMENT2_DIR"], "1.ply")
    print(f"Processing {pc_path}")
    pcd = o3d.io.read_point_cloud(pc_path)
    print("read point cloud")
    vis = o3d.visualization.VisualizerWithEditing()
    print("created visualizer")
    vis.create_window()
    print("created window")
    vis.add_geometry(pcd)
    print("added geometry")
    vis.run()  # User picks points
    print("ran")
    vis.destroy_window()
    print("destroyed window")
    picked_points = vis.get_picked_points()
    print(f"Selected points indices: {picked_points}")
    np.savetxt(
        os.path.join(os.environ["EXPERIMENT2_DIR"], "selected_indices.txt"),
        picked_points,
        fmt="%d",
    )
