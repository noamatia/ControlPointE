import os
from point_e.util.point_cloud import PointCloud


DATA_DIR = "data"
SOURCE_UID = "source_uid"
TARGET_UID = "target_uid"
PARTNET_METADATA_DIR = "partnet"
PARTNET_DIR = "/scratch/noam/data_v0"
PARTNET_METADATA_JSON = "partnet_metadata.json"
PCS_DIR = "/scratch/noam/shapetalk/point_clouds/scaled_to_align_rendering"


def load_partnet(src_dir, shapenet_uid, masked_labels):
    return PointCloud.load_partnet(
        os.path.join(
            src_dir, "point_sample", "sample-points-all-pts-nor-rgba-10000.txt"
        ),
        os.path.join(src_dir, "point_sample", "sample-points-all-label-10000.txt"),
        os.path.join(PCS_DIR, shapenet_uid + ".npz"),
        masked_labels,
    )
