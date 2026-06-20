"""Load KITTI ground-truth camera poses from a text file."""

from pathlib import Path

import numpy as np


def read_ground_truth_poses(poses_path: Path) -> np.ndarray:
    """Reads KITTI-style 3x4 ground-truth camera extrinsic matrices."""
    poses = []

    with open(poses_path) as f:
        for line in f:
            pose = np.array([float(x) for x in line.split()]).reshape(3, 4)
            poses.append(pose)

    return np.asarray(poses)
