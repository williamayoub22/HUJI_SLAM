"""Triangulate 3D points from calibrated stereo correspondences."""

import cv2
import numpy as np


def _validate_triangulation_inputs(
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize stereo triangulation inputs.

    Args:
        projection_left: Left-camera projection matrix.
        projection_right: Right-camera projection matrix.
        left_points: Left-image points with shape ``(N, 2)``.
        right_points: Right-image points with shape ``(N, 2)``.

    Returns:
        Float-valued validated projection matrices and image-point arrays.

    Raises:
        ValueError: If projection matrices or point arrays have invalid shapes.
    """
    projection_left = np.asarray(projection_left, dtype=float)
    projection_right = np.asarray(projection_right, dtype=float)
    left_points = np.asarray(left_points, dtype=float)
    right_points = np.asarray(right_points, dtype=float)

    if projection_left.shape != (3, 4):
        raise ValueError(f"projection_left must have shape (3, 4), got {projection_left.shape}.")

    if projection_right.shape != (3, 4):
        raise ValueError(f"projection_right must have shape (3, 4), got {projection_right.shape}.")

    if left_points.ndim != 2 or left_points.shape[1] != 2:
        raise ValueError(f"left_points must have shape (N, 2), got {left_points.shape}.")

    if right_points.shape != left_points.shape:
        raise ValueError(
            "right_points must match left_points shape, got "
            f"{right_points.shape} and {left_points.shape}."
        )

    return projection_left, projection_right, left_points, right_points


def triangulate_dlt(
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> np.ndarray:
    """Triangulate stereo correspondences using linear DLT.

    Args:
        projection_left: Left-camera projection matrix with shape ``(3, 4)``.
        projection_right: Right-camera projection matrix with shape ``(3, 4)``.
        left_points: Left-image coordinates with shape ``(N, 2)``.
        right_points: Right-image coordinates with shape ``(N, 2)``.

    Returns:
        Triangulated 3D points in the shared camera coordinate system, with
        shape ``(N, 3)``.

    Raises:
        ValueError: If inputs have invalid shapes or a point is at infinity.
    """
    projection_left, projection_right, left_points, right_points = _validate_triangulation_inputs(
        projection_left,
        projection_right,
        left_points,
        right_points,
    )

    num_points = len(left_points)
    points_3d = np.empty((num_points, 3), dtype=float)

    for index, (left_point, right_point) in enumerate(zip(left_points, right_points, strict=True)):
        u_left, v_left = left_point
        u_right, v_right = right_point

        design_matrix = np.vstack(
            (
                u_left * projection_left[2] - projection_left[0],
                v_left * projection_left[2] - projection_left[1],
                u_right * projection_right[2] - projection_right[0],
                v_right * projection_right[2] - projection_right[1],
            )
        )

        _, _, right_singular_vectors = np.linalg.svd(design_matrix)
        point_homogeneous = right_singular_vectors[-1]

        if np.isclose(point_homogeneous[3], 0.0):
            raise ValueError(f"Triangulation produced a point at infinity at index {index}.")

        points_3d[index] = point_homogeneous[:3] / point_homogeneous[3]

    return points_3d


def triangulate_opencv(
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> np.ndarray:
    """Triangulate stereo correspondences using OpenCV.

    Args:
        projection_left: Left-camera projection matrix with shape ``(3, 4)``.
        projection_right: Right-camera projection matrix with shape ``(3, 4)``.
        left_points: Left-image coordinates with shape ``(N, 2)``.
        right_points: Right-image coordinates with shape ``(N, 2)``.

    Returns:
        Triangulated 3D points in the shared camera coordinate system, with
        shape ``(N, 3)``.

    Raises:
        ValueError: If inputs have invalid shapes or a point is at infinity.
    """
    projection_left, projection_right, left_points, right_points = _validate_triangulation_inputs(
        projection_left,
        projection_right,
        left_points,
        right_points,
    )

    if len(left_points) == 0:
        return np.empty((0, 3), dtype=float)

    points_homogeneous = cv2.triangulatePoints(
        projection_left,
        projection_right,
        left_points.T,
        right_points.T,
    )

    scales = points_homogeneous[3]

    if np.any(np.isclose(scales, 0.0)):
        raise ValueError("OpenCV triangulation produced a point at infinity.")

    return (points_homogeneous[:3] / scales).T
