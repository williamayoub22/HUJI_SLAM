from pathlib import Path
from typing import Sequence

import numpy as np


def to_homogeneous_transform(T: np.ndarray) -> np.ndarray:
    """Converts either a 3x4 [R|t] matrix or a 4x4 matrix into a 4x4 transform."""
    if T.shape == (4, 4):
        return T

    if T.shape == (3, 4):
        T_hom = np.eye(4)
        T_hom[:3, :] = T
        return T_hom

    raise ValueError(f"Unexpected transformation shape: {T.shape}")


def compose_global_camera_matrices(
    relative_transforms: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Composes consecutive PnP relative transforms into global camera matrices.

    Parameters
    ----------
    relative_transforms:
        relative_transforms[i] is assumed to be T_{i -> i+1}, i.e.
        it maps points from camera i coordinates to camera i+1 coordinates.

    Returns
    -------
    global_camera_matrices:
        Array of shape (N, 4, 4), where N = len(relative_transforms) + 1.
        global_camera_matrices[i] is T_{0 -> i}, mapping points from
        frame-0 coordinates to frame-i coordinates.

    Notes
    -----
    The first frame is used as the global coordinate system, therefore
    T_{0 -> 0} is the identity matrix.
    """
    global_matrices = [np.eye(4)]

    T_0_to_current = np.eye(4)

    for T_current_to_next in relative_transforms:
        T_current_to_next = to_homogeneous_transform(T_current_to_next)
        T_0_to_current = T_current_to_next @ T_0_to_current
        global_matrices.append(T_0_to_current.copy())

    return np.asarray(global_matrices)


def camera_center_from_extrinsic(T_0_to_i: np.ndarray) -> np.ndarray:
    """
    Computes the camera center of frame i in the frame-0/global coordinate system.
    If the global extrinsic is:
        x_i = R x_0 + t

    then the camera center C_i is the point satisfying x_i = 0:
        C_i = -R^T t
    """
    T_0_to_i = to_homogeneous_transform(T_0_to_i)
    R = T_0_to_i[:3, :3]
    t = T_0_to_i[:3, 3]
    return -R.T @ t


def camera_centers_from_extrinsics(global_camera_matrices: np.ndarray) -> np.ndarray:
    """
    Computes all camera centers in the frame-0/global coordinate system.
    """
    return np.asarray([
        camera_center_from_extrinsic(T)
        for T in global_camera_matrices
    ])


def save_global_camera_matrices(
    relative_transforms: Sequence[np.ndarray],
    output_dir: str | Path,
) -> np.ndarray:
    """
    Composes and saves global PnP camera matrices.

    Saves:
        global_camera_matrices.npy
        camera_centers.npy

    Returns
    -------
    global_camera_matrices:
        The saved array, with shape (N, 4, 4).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_camera_matrices = compose_global_camera_matrices(relative_transforms)
    camera_centers = camera_centers_from_extrinsics(global_camera_matrices)
    np.save(output_dir / "global_camera_matrices.npy", global_camera_matrices)
    np.save(output_dir / "camera_centers.npy", camera_centers)
    return global_camera_matrices


def load_global_camera_matrices(path: str | Path) -> np.ndarray:
    """
    Loads global camera matrices saved by save_global_camera_matrices().
    """
    return np.load(path)

def relative_extrinsic(
    T_global_to_ref: np.ndarray,
    T_global_to_frame: np.ndarray,
) -> np.ndarray:
    """
    Converts a global extrinsic T_{0 -> i} to a local-window extrinsic T_{s -> i}.

    Both input matrices are assumed to map from the global/frame-0 coordinate
    system to the corresponding camera coordinate system.
    """
    return T_global_to_frame @ np.linalg.inv(T_global_to_ref)