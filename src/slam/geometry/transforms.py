"""Coordinate-transform utilities for camera extrinsics and trajectories."""

from collections.abc import Sequence

import numpy as np


def to_homogeneous_transform(transform: np.ndarray) -> np.ndarray:
    """Convert a ``(3, 4)`` or ``(4, 4)`` transform to homogeneous form.

    Args:
        transform: Rotation-translation matrix with shape ``(3, 4)`` or an
            already homogeneous transform with shape ``(4, 4)``.

    Returns:
        Homogeneous transform with shape ``(4, 4)``.

    Raises:
        ValueError: If ``transform`` has an unsupported shape.
    """
    transform = np.asarray(transform, dtype=float)

    if transform.shape == (4, 4):
        return transform

    if transform.shape == (3, 4):
        homogeneous_transform = np.eye(4)
        homogeneous_transform[:3, :] = transform
        return homogeneous_transform

    raise ValueError(f"Expected a transform with shape (3, 4) or (4, 4), got {transform.shape}.")


def camera_center_from_world_to_camera_extrinsic(
    world_to_camera: np.ndarray,
) -> np.ndarray:
    """Return the camera center in world coordinates from a world-to-camera map.

    The extrinsic follows:

    ``x_camera = R @ x_world + t``.

    Therefore, the camera center in world coordinates is ``-R.T @ t``.

    Args:
        world_to_camera: World-to-camera transform with shape ``(3, 4)`` or
            ``(4, 4)``.

    Returns:
        Camera center with shape ``(3,)`` in world coordinates.
    """
    world_to_camera = to_homogeneous_transform(world_to_camera)

    rotation_world_to_camera = world_to_camera[:3, :3]
    translation_world_to_camera = world_to_camera[:3, 3]

    return -rotation_world_to_camera.T @ translation_world_to_camera


def camera_centers_from_world_to_camera_extrinsics(
    world_to_camera_extrinsics: np.ndarray,
) -> np.ndarray:
    """Return camera centers in world coordinates for multiple extrinsics.

    Args:
        world_to_camera_extrinsics: Transform array with shape ``(N, 3, 4)``
            or ``(N, 4, 4)``.

    Returns:
        Camera centers with shape ``(N, 3)``.
    """
    return np.asarray(
        [
            camera_center_from_world_to_camera_extrinsic(extrinsic)
            for extrinsic in world_to_camera_extrinsics
        ],
        dtype=float,
    )


def relative_world_to_camera_extrinsic(
    world_to_reference: np.ndarray,
    world_to_frame: np.ndarray,
) -> np.ndarray:
    """Express a frame's world-to-camera extrinsic in reference-camera coordinates.

    Given two transforms:

    ``x_reference = T_world_to_reference @ x_world``

    ``x_frame = T_world_to_frame @ x_world``

    this returns:

    ``x_frame = T_reference_to_frame @ x_reference``.

    Args:
        world_to_reference: World-to-reference-camera extrinsic.
        world_to_frame: World-to-frame-camera extrinsic.

    Returns:
        Reference-camera-to-frame-camera transform with shape ``(4, 4)``.
    """
    world_to_reference = to_homogeneous_transform(world_to_reference)
    world_to_frame = to_homogeneous_transform(world_to_frame)

    return world_to_frame @ np.linalg.inv(world_to_reference)


def get_relative_camera_transform(world_to_camera_by_frame, source_frame, target_frame):
    """
    Return the estimated transform from source-camera coordinates to
    target-camera coordinates.
    """
    return relative_world_to_camera_extrinsic(
        world_to_camera_by_frame[source_frame],
        world_to_camera_by_frame[target_frame],
    )


def compose_camera_transform(world_to_camera_by_frame, source_frame, target_frame):
    """
    Compose all consecutive estimated camera transforms from source_frame
    to target_frame.

    The returned transform maps a point from source-camera coordinates into
    target-camera coordinates.
    """
    if source_frame == target_frame:
        return np.eye(4, dtype=float)
    step = 1 if target_frame > source_frame else -1
    current_frame = source_frame
    target_from_source = np.eye(4, dtype=float)
    while current_frame != target_frame:
        next_frame = current_frame + step
        if (
            current_frame not in world_to_camera_by_frame
            or next_frame not in world_to_camera_by_frame
        ):
            raise KeyError(
                f"Missing estimated camera pose for transition {current_frame} -> {next_frame}"
            )
        next_from_current = get_relative_camera_transform(
            world_to_camera_by_frame=world_to_camera_by_frame,
            source_frame=current_frame,
            target_frame=next_frame,
        )
        target_from_source = next_from_current @ target_from_source
        current_frame = next_frame
    return target_from_source
