from dataclasses import dataclass

import cv2
import numpy as np

from ..features.matching import get_matched_points
from ..geometry.triangulation import triangulate_dlt, triangulate_opencv
from ..io.calibration import read_stereo_calibration
from ..io.image_loader import load_matches_between_images

DEVIATION_THRESHOLD = 2.0
MAX_DEPTH = 300.0


@dataclass
class StereoMatchData:
    """Container for one stereo pair, its matches, and vertical deviations."""

    left_img: np.ndarray
    right_img: np.ndarray
    kp_left: list[cv2.KeyPoint]
    kp_right: list[cv2.KeyPoint]
    matches: list[cv2.DMatch]
    left_pts: np.ndarray
    right_pts: np.ndarray
    deviations: np.ndarray


@dataclass
class StereoPointCloud:
    """Container for a stereo pair and its triangulated 3D point cloud."""

    frame_idx: int
    data: StereoMatchData
    left_inliers: np.ndarray
    right_inliers: np.ndarray
    points_3d: np.ndarray


def load_frame_data(
    frame_idx: int,
    feature_type: str = "sift",
    num_features: int = 1000,
    use_ratio_test: bool = False,
) -> StereoMatchData:
    """Loads one stereo frame and precomputes matched points and deviations."""
    left_keypoints, right_keypoints, left_image, right_image, matches = load_matches_between_images(
        frame_idx,
        feature_type=feature_type,
        num_features=num_features,
        use_ratio_test=use_ratio_test,
    )
    left_pts, right_pts = get_matched_points(left_keypoints, right_keypoints, matches)
    deviations = np.abs(left_pts[:, 1] - right_pts[:, 1])

    return StereoMatchData(
        left_img=left_image,
        right_img=right_image,
        kp_left=left_keypoints,
        kp_right=right_keypoints,
        matches=matches,
        left_pts=left_pts,
        right_pts=right_pts,
        deviations=deviations,
    )


def get_inlier_points(
    data: StereoMatchData,
    threshold: float = DEVIATION_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns matched points that satisfy the vertical-deviation threshold."""
    inlier_mask = data.deviations <= threshold
    return data.left_pts[inlier_mask], data.right_pts[inlier_mask]


def keep_valid_depth_points(
    points_3d: np.ndarray,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    max_depth: float = MAX_DEPTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keeps only triangulated points with positive depth and within max_depth.
    Points with z <= 0 are behind the camera; points with z >= max_depth
    amplify noise and hurt PnP estimation.
    """
    valid_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < max_depth)

    return (
        points_3d[valid_mask],
        left_pts[valid_mask],
        right_pts[valid_mask],
    )


def create_stereo_point_cloud(
    frame_idx: int,
    threshold: float = DEVIATION_THRESHOLD,
    use_custom_triangulation: bool = False,
    reject_negative_depth: bool = False,
    max_depth: float = MAX_DEPTH,
    feature_type: str = "sift",
    num_features: int = 1000,
    use_ratio_test: bool = False,
) -> StereoPointCloud:
    """Creates a 3D point cloud for one stereo pair.

    Pipeline:
    1. Load and match the left/right stereo images.
    2. Reject matches whose vertical deviation is too large.
    3. Triangulate the remaining matches.
    4. Optionally reject triangulated points with non-positive depth or beyond max_depth.
    """
    P1, P2 = read_stereo_calibration()

    data = load_frame_data(
        frame_idx,
        feature_type=feature_type,
        num_features=num_features,
        use_ratio_test=use_ratio_test,
    )
    left_inliers, right_inliers = get_inlier_points(data, threshold)

    if use_custom_triangulation:
        points_3d = triangulate_dlt(P1, P2, left_inliers, right_inliers)
    else:
        points_3d = triangulate_opencv(P1, P2, left_inliers, right_inliers)

    if reject_negative_depth:
        points_3d, left_inliers, right_inliers = keep_valid_depth_points(
            points_3d,
            left_inliers,
            right_inliers,
            max_depth,
        )

    return StereoPointCloud(
        frame_idx=frame_idx,
        data=data,
        left_inliers=left_inliers,
        right_inliers=right_inliers,
        points_3d=points_3d,
    )


def compute_rejection_statistics(
    data: StereoMatchData,
    threshold: float = DEVIATION_THRESHOLD,
) -> tuple[int, int, float]:
    """Computes rejection statistics for a vertical-deviation threshold."""
    num_matches = len(data.matches)
    num_rejected = int(np.sum(data.deviations > threshold))
    percentage_rejected = 100.0 * num_rejected / num_matches

    return num_matches, num_rejected, percentage_rejected
