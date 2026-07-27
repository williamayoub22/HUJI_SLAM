"""Projection and four-view reprojection-consistency utilities."""

from dataclasses import dataclass

import numpy as np

from .. import config


@dataclass(frozen=True)
class FourViewReprojectionErrors:
    """Store reprojection errors for two consecutive stereo pairs.

    Attributes:
        left0: Errors in the first left image, shape ``(N,)``.
        right0: Errors in the first right image, shape ``(N,)``.
        left1: Errors in the second left image, shape ``(N,)``.
        right1: Errors in the second right image, shape ``(N,)``.
    """

    left0: np.ndarray
    right0: np.ndarray
    left1: np.ndarray
    right1: np.ndarray

    def as_array(self) -> np.ndarray:
        """Return errors as an ``(N, 4)`` array in left0/right0/left1/right1 order."""
        return np.column_stack((self.left0, self.right0, self.left1, self.right1))


def project_points(
    points_3d: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    """Project 3D points through a pinhole camera projection matrix.

    Args:
        points_3d: Euclidean 3D points with shape ``(N, 3)``.
        projection_matrix: Camera projection matrix with shape ``(3, 4)``.

    Returns:
        Projected image coordinates with shape ``(N, 2)``.

    Raises:
        ValueError: If the inputs have incompatible shapes or a point projects
            with zero depth.
    """
    points_3d = np.asarray(points_3d, dtype=float)
    projection_matrix = np.asarray(projection_matrix, dtype=float)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N, 3), got {points_3d.shape}.")

    if projection_matrix.shape != (3, 4):
        raise ValueError(
            f"projection_matrix must have shape (3, 4), got {projection_matrix.shape}."
        )

    points_homogeneous = np.column_stack((points_3d, np.ones(len(points_3d), dtype=float)))
    projected_homogeneous = (projection_matrix @ points_homogeneous.T).T
    depths = projected_homogeneous[:, 2]

    if np.any(np.isclose(depths, 0.0)):
        raise ValueError("Cannot project points with zero depth.")

    return projected_homogeneous[:, :2] / depths[:, None]


def four_view_reprojection_errors(
    transform_left0_to_left1: np.ndarray,
    points_3d_left0: np.ndarray,
    left0: np.ndarray,
    right0: np.ndarray,
    left1: np.ndarray,
    right1: np.ndarray,
    intrinsic_matrix: np.ndarray,
    left_projection_matrix: np.ndarray,
    right_projection_matrix: np.ndarray,
) -> FourViewReprojectionErrors:
    """Compute reprojection errors in two consecutive rectified stereo pairs.

    ``transform_left0_to_left1`` maps points from the first left-camera frame
    into the second left-camera frame.

    Args:
        transform_left0_to_left1: Extrinsic transform ``[R | t]`` with shape
            ``(3, 4)``.
        points_3d_left0: 3D points expressed in the first left-camera frame.
        left0: Observations in the first left image.
        right0: Observations in the first right image.
        left1: Observations in the second left image.
        right1: Observations in the second right image.
        intrinsic_matrix: First-left camera intrinsic matrix, shape ``(3, 3)``.
        left_projection_matrix: First-left projection matrix, shape ``(3, 4)``.
        right_projection_matrix: First-right projection matrix, shape ``(3, 4)``.

    Returns:
        Per-point Euclidean reprojection errors in all four images.
    """
    transform_left0_to_left1 = np.asarray(transform_left0_to_left1, dtype=float)

    if transform_left0_to_left1.shape != (3, 4):
        raise ValueError(
            "transform_left0_to_left1 must have shape (3, 4), "
            f"got {transform_left0_to_left1.shape}."
        )

    left1_projection = intrinsic_matrix @ transform_left0_to_left1

    transform_left0_to_left1_homogeneous = np.vstack(
        (transform_left0_to_left1, np.array([0.0, 0.0, 0.0, 1.0]))
    )
    right1_projection = right_projection_matrix @ transform_left0_to_left1_homogeneous

    return FourViewReprojectionErrors(
        left0=np.linalg.norm(
            project_points(points_3d_left0, left_projection_matrix) - left0,
            axis=1,
        ),
        right0=np.linalg.norm(
            project_points(points_3d_left0, right_projection_matrix) - right0,
            axis=1,
        ),
        left1=np.linalg.norm(
            project_points(points_3d_left0, left1_projection) - left1,
            axis=1,
        ),
        right1=np.linalg.norm(
            project_points(points_3d_left0, right1_projection) - right1,
            axis=1,
        ),
    )


def count_supporters(
    transform_left0_to_left1: np.ndarray,
    points_3d_left0: np.ndarray,
    left0: np.ndarray,
    right0: np.ndarray,
    left1: np.ndarray,
    right1: np.ndarray,
    intrinsic_matrix: np.ndarray,
    left_projection_matrix: np.ndarray,
    right_projection_matrix: np.ndarray,
    threshold_pixels: float = config.SUPPORTER_THRESHOLD_PIXELS,
) -> tuple[np.ndarray, FourViewReprojectionErrors]:
    """Return correspondences consistent with a four-view reprojection threshold.

    A correspondence is a supporter only when its reprojection error is below
    ``threshold_pixels`` in all four images.

    Args:
        transform_left0_to_left1: Transform from left camera 0 to left camera 1.
        points_3d_left0: 3D points in left-camera-0 coordinates.
        left0: Observations in the first left image.
        right0: Observations in the first right image.
        left1: Observations in the second left image.
        right1: Observations in the second right image.
        intrinsic_matrix: Left-camera intrinsic matrix.
        left_projection_matrix: First-left projection matrix.
        right_projection_matrix: First-right projection matrix.
        threshold_pixels: Maximum allowed reprojection error per image.

    Returns:
        Boolean supporter mask and per-image reprojection errors.

    Raises:
        ValueError: If ``threshold_pixels`` is not positive.
    """
    if threshold_pixels <= 0:
        raise ValueError("threshold_pixels must be positive.")

    errors = four_view_reprojection_errors(
        transform_left0_to_left1=transform_left0_to_left1,
        points_3d_left0=points_3d_left0,
        left0=left0,
        right0=right0,
        left1=left1,
        right1=right1,
        intrinsic_matrix=intrinsic_matrix,
        left_projection_matrix=left_projection_matrix,
        right_projection_matrix=right_projection_matrix,
    )

    supporter_mask = np.all(
        errors.as_array() < threshold_pixels,
        axis=1,
    )

    return supporter_mask, errors


import cv2

def solve_pnp_safe(pts_3d, pts_2d, K, flags):
    try:
        success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, None, flags=flags)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.hstack((R, tvec))
            return T
    except cv2.error:
        pass
    return None
