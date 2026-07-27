"""Build four-view stereo-temporal correspondences from tracked image points."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FourViewCorrespondences:
    """Matched observations of landmarks across two consecutive stereo pairs.

    Attributes:
        points_3d: 3D points triangulated from the first stereo pair, shape
            ``(N, 3)``.
        left0: Pixel coordinates in the first left image, shape ``(N, 2)``.
        right0: Pixel coordinates in the first right image, shape ``(N, 2)``.
        left1: Pixel coordinates in the second left image, shape ``(N, 2)``.
        right1: Pixel coordinates in the second right image, shape ``(N, 2)``.
    """

    points_3d: np.ndarray
    left0: np.ndarray
    right0: np.ndarray
    left1: np.ndarray
    right1: np.ndarray


def find_common_points(
    points_3d0: np.ndarray,
    left_inliers0: np.ndarray,
    right_inliers0: np.ndarray,
    left_inliers1: np.ndarray,
    right_inliers1: np.ndarray,
    temporal_left0: np.ndarray,
    temporal_left1: np.ndarray,
    tolerance: float = 1e-3,
) -> FourViewCorrespondences:
    """Find points observed in two consecutive stereo pairs.

    Each temporal correspondence links a left-image point in frame 0 to one in
    frame 1. The function associates both points with their corresponding stereo
    inliers, producing observations in left0, right0, left1, and right1.

    Args:
        points_3d0: 3D points for frame 0, shape (N, 3).
        left_inliers0: Left-image stereo inliers for frame 0, shape (N, 2).
        right_inliers0: Right-image stereo inliers for frame 0, shape (N, 2).
        left_inliers1: Left-image stereo inliers for frame 1, shape (N, 2).
        right_inliers1: Right-image stereo inliers for frame 1, shape (N, 2).
        temporal_left0: Matched left-image points in frame 0, shape ``(N, 2)``.
        temporal_left1: Corresponding left-image points in frame 1, shape
            ``(N, 2)``.
        tolerance: Maximum Euclidean pixel distance used to associate a temporal
            point with a stereo inlier.

    Returns:
        Four-view correspondences and 3D points from the first stereo pair.

    Raises:
        ValueError: If temporal point arrays are misaligned or tolerance is
            negative.
    """
    if temporal_left0.shape != temporal_left1.shape:
        raise ValueError(
            "Temporal point arrays must have identical shapes, got "
            f"{temporal_left0.shape} and {temporal_left1.shape}."
        )

    if temporal_left0.ndim != 2 or temporal_left0.shape[1] != 2:
        raise ValueError(
            f"Temporal point arrays must have shape (N, 2), got {temporal_left0.shape}."
        )

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    points_3d: list[np.ndarray] = []
    left0_points: list[np.ndarray] = []
    right0_points: list[np.ndarray] = []
    left1_points: list[np.ndarray] = []
    right1_points: list[np.ndarray] = []

    for left0_point, left1_point in zip(
        temporal_left0,
        temporal_left1,
        strict=True,
    ):
        distances0 = np.linalg.norm(
            left_inliers0 - left0_point,
            axis=1,
        )
        index0 = int(np.argmin(distances0))

        distances1 = np.linalg.norm(
            left_inliers1 - left1_point,
            axis=1,
        )
        index1 = int(np.argmin(distances1))

        if distances0[index0] > tolerance or distances1[index1] > tolerance:
            continue

        points_3d.append(points_3d0[index0])
        left0_points.append(left_inliers0[index0])
        right0_points.append(right_inliers0[index0])
        left1_points.append(left_inliers1[index1])
        right1_points.append(right_inliers1[index1])

    if not points_3d:
        return FourViewCorrespondences(
            points_3d=np.empty((0, 3), dtype=np.float64),
            left0=np.empty((0, 2), dtype=np.float64),
            right0=np.empty((0, 2), dtype=np.float64),
            left1=np.empty((0, 2), dtype=np.float64),
            right1=np.empty((0, 2), dtype=np.float64),
        )

    return FourViewCorrespondences(
        points_3d=np.asarray(points_3d, dtype=np.float64),
        left0=np.asarray(left0_points, dtype=np.float64),
        right0=np.asarray(right0_points, dtype=np.float64),
        left1=np.asarray(left1_points, dtype=np.float64),
        right1=np.asarray(right1_points, dtype=np.float64),
    )
