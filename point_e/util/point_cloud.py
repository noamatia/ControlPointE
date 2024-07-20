import random
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Union

import torch
import numpy as np

from .ply_util import write_ply

COLORS = frozenset(["R", "G", "B", "A"])


def preprocess(data, channel):
    if channel in COLORS:
        return np.round(data * 255.0)
    return data


@dataclass
class PointCloud:
    """
    An array of points sampled on a surface. Each point may have zero or more
    channel attributes.

    :param coords: an [N x 3] array of point coordinates.
    :param channels: a dict mapping names to [N] arrays of channel values.
    """

    coords: np.ndarray
    channels: Dict[str, np.ndarray]
    mask: Optional[np.ndarray] = None

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return PointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"},
            )
        
    @classmethod
    def load_shapenet(cls, path: str) -> "PointCloud":
        """
        Load the shapebet point cloud from a .npz file.
        """
        with open(path, "rb") as fn:
            coords = np.load(fn)["pointcloud"].astype(np.float32)
        coords[:, [0, 1, 2]] = coords[:, [2, 0, 1]]
        channels = {k: np.zeros_like(coords[:, 0], dtype=np.float32) for k in ["R", "G", "B"]}
        return PointCloud(
            coords=coords,
            channels=channels,
        )
        
    @classmethod
    def load_partnet(cls, path: str, labels_path: str, shapenet_path: str, masked_labels: list) -> "PointCloud":
        """
        Load the partnet point cloud from a .txt file. 
        """
        with open(path, "r") as fin:
            lines = [line.rstrip().split() for line in fin.readlines()]
        coords = np.array([[float(line[0]), float(line[1]), float(line[2])] for line in lines], dtype=np.float32)
        coords[:, [0, 1, 2]] = coords[:, [2, 0, 1]]
        coords = cls.normalize_coords(coords, shapenet_path)
        channels = {k: np.zeros_like(coords[:, 0], dtype=np.float32) for k in "RGB"}
        mask = None
        with open(labels_path, "r") as fin:
            labels = np.array([int(item.rstrip()) for item in fin.readlines()], dtype=np.int32)
        mask = np.isin(labels, masked_labels)
        mask = 1 - mask.astype(int)
        return PointCloud(
            coords=coords,
            channels=channels,
            mask=mask
        )
    
    @classmethod
    def normalize_coords(cls, partnet_coords, shapenet_path):
        shapenet_coords = cls.load_shapenet(shapenet_path).coords
        partnet_min = np.min(partnet_coords, axis=0)
        partnet_max = np.max(partnet_coords, axis=0)
        shapenet_min = np.min(shapenet_coords, axis=0)
        shapenet_max = np.max(shapenet_coords, axis=0)
        partnet_coords = (partnet_coords - partnet_min) / (partnet_max - partnet_min) * (shapenet_max - shapenet_min) + shapenet_min
        return partnet_coords

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with open(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.coords,
            rgb=(
                np.stack([self.channels[x] for x in "RGB"], axis=1)
                if all(x in self.channels for x in "RGB")
                else None
            ),
        )

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx
            cur_dists = np.minimum(cur_dists, compute_dists(idx))
        return self.subsample(indices, **subsample_kwargs)

    def subsample(self, indices: np.ndarray, average_neighbors: bool = False) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
                mask=self.mask[indices] if self.mask is not None else None,
            )

        new_coords = self.coords[indices]
        neighbor_indices = PointCloud(coords=new_coords, channels={}).nearest_points(self.coords)

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(coords=new_coords, channels=new_channels)

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack([preprocess(self.channels[name], name) for name in channel_names], axis=-1)
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            dists = norms + np.sum(batch**2, axis=-1)[:, None] - 2 * (batch @ self.coords.T)
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0) for k, v in self.channels.items()
            },
        )

    def encode(self) -> torch.Tensor:
        """
        Encode the point cloud to a Kx6 tensor where K is the number of points.
        """
        coords = torch.tensor(self.coords.T, dtype=torch.float32)
        rgb = [(self.channels[x] * 255).astype(np.uint8) for x in "RGB"]
        rgb = [torch.tensor(x, dtype=torch.float32) for x in rgb]
        rgb = torch.stack(rgb, dim=0)
        return torch.cat([coords, rgb], dim=0)
    
    def encode_mask(self) -> torch.Tensor:
        """
        Encode the mask to a Kx6 tensor where K is the number of points.
        """
        return torch.tensor(np.tile(self.mask, (6, 1)), dtype=torch.float32)
    