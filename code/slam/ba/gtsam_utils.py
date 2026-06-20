"""Adapters between KITTI-style stereo data, NumPy arrays, and GTSAM types."""

import gtsam
import numpy as np


def stereo_image_distances(
    measurement: gtsam.StereoPoint2,
    projection: gtsam.StereoPoint2,
) -> tuple[float, float]:
    """Return left and right Euclidean pixel distances between stereo points."""
    left_distance = np.hypot(
        projection.uL() - measurement.uL(),
        projection.v() - measurement.v(),
    )
    right_distance = np.hypot(
        projection.uR() - measurement.uR(),
        projection.v() - measurement.v(),
    )
    return float(left_distance), float(right_distance)


def make_gtsam_stereo_calibration(
    P1: np.ndarray,
    P2: np.ndarray,
) -> gtsam.Cal3_S2Stereo:
    """Create GTSAM stereo calibration from rectified KITTI projection matrices.

    Assumes ``P2[0, 3] = -fx * baseline``.

    Args:
        P1: Left-camera rectified projection matrix.
        P2: Right-camera rectified projection matrix.

    Returns:
        GTSAM stereo calibration containing intrinsics and baseline.

    Raises:
        ValueError: If the focal length is zero.
    """
    fx = float(P1[0, 0])
    if fx == 0:
        raise ValueError("Cannot infer stereo baseline because fx is zero.")

    fy = float(P1[1, 1])
    skew = float(P1[0, 1])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    baseline = float(-P2[0, 3] / fx)

    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)


def pose3_from_world_to_camera_extrinsic(
    T_world_to_camera: np.ndarray,
) -> gtsam.Pose3:
    """Convert a world-to-camera extrinsic matrix to a camera-to-world Pose3.

    The input follows ``x_camera = R @ x_world + t``. GTSAM's ``Pose3``
    represents the inverse transform, mapping camera coordinates into world
    coordinates.

    Args:
        T_world_to_camera: Extrinsic matrix with shape ``(3, 4)`` or ``(4, 4)``.

    Returns:
        Camera pose that maps camera coordinates into world coordinates.

    Raises:
        ValueError: If the matrix does not have shape ``(3, 4)`` or ``(4, 4)``.
    """
    if T_world_to_camera.shape not in {(3, 4), (4, 4)}:
        raise ValueError(
            "Expected a transform with shape (3, 4) or (4, 4), "
            f"got {T_world_to_camera.shape}."
        )

    R_world_to_camera = T_world_to_camera[:3, :3]
    t_world_to_camera = T_world_to_camera[:3, 3]

    R_camera_to_world = R_world_to_camera.T
    t_camera_to_world = -R_camera_to_world @ t_world_to_camera

    return gtsam.Pose3(
        gtsam.Rot3(R_camera_to_world),
        gtsam.Point3(t_camera_to_world),
    )


def make_stereo_camera(
    T_world_to_camera: np.ndarray,
    calibration: gtsam.Cal3_S2Stereo,
) -> gtsam.StereoCamera:
    """Create a GTSAM stereo camera from a world-to-camera extrinsic matrix.

    Args:
        T_world_to_camera: World-to-camera extrinsic matrix.
        calibration: Stereo camera calibration.

    Returns:
        GTSAM stereo camera whose pose maps camera coordinates into world coordinates.
    """
    pose = pose3_from_world_to_camera_extrinsic(T_world_to_camera)
    return gtsam.StereoCamera(pose, calibration)


def stereo_point_from_triplet(
    triplet: tuple[float, float, float],
) -> gtsam.StereoPoint2:
    """Convert a tracking triplet ``(u_left, u_right, v)`` to StereoPoint2."""
    u_left, u_right, v = triplet
    return gtsam.StereoPoint2(u_left, u_right, v)


def stereo_point_to_array(point: gtsam.StereoPoint2) -> np.ndarray:
    """Convert a StereoPoint2 to ``[u_left, u_right, v]``."""
    return np.array([point.uL(), point.uR(), point.v()], dtype=float)

def stereo_residual_norm(
    measurement: gtsam.StereoPoint2,
    projection: gtsam.StereoPoint2,
) -> float:
    """Return the L2 norm of a three-coordinate stereo measurement residual.

    The residual is ordered as ``[u_left, u_right, v]``.

    Args:
        measurement: Observed stereo image measurement.
        projection: Predicted stereo image measurement.

    Returns:
        Euclidean norm of the three-coordinate stereo residual.
    """
    residual = stereo_point_to_array(measurement) - stereo_point_to_array(projection)
    return float(np.linalg.norm(residual))

