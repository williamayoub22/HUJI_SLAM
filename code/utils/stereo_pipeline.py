from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from utils.image_loader import load_matches_between_images
from utils.matching import get_matched_points
from utils.read_cam_calib import read_calib
from utils.triangulation import custom_triangulation, triangulate_opencv


DEVIATION_THRESHOLD = 2.0


@dataclass
class StereoMatchData:
    """Container for one stereo pair, its matches, and vertical deviations."""
    left_img: np.ndarray
    right_img: np.ndarray
    kp_left: List[cv2.KeyPoint]
    kp_right: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]
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


def load_frame_data(frame_idx: int) -> StereoMatchData:
    """Loads one stereo frame and precomputes matched points and deviations."""
    kp_left, kp_right, left_img, matches, right_img = load_matches_between_images(frame_idx)
    left_pts, right_pts = get_matched_points(kp_left, kp_right, matches)
    deviations = np.abs(left_pts[:, 1] - right_pts[:, 1])

    return StereoMatchData(
        left_img=left_img,
        right_img=right_img,
        kp_left=kp_left,
        kp_right=kp_right,
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


def create_stereo_point_cloud(
    frame_idx: int,
    threshold: float = DEVIATION_THRESHOLD,
    use_custom_triangulation: bool = False,
) -> StereoPointCloud:
    """
    Creates a 3D point cloud for one stereo pair.

    Pipeline:
    1. Load and match the left/right stereo images.
    2. Reject matches whose vertical deviation is too large.
    3. Triangulate the remaining matches.
    """
    P1, P2 = read_calib()

    data = load_frame_data(frame_idx)
    left_inliers, right_inliers = get_inlier_points(data, threshold)

    if use_custom_triangulation:
        points_3d = custom_triangulation(P1, P2, left_inliers, right_inliers)
    else:
        points_3d = triangulate_opencv(P1, P2, left_inliers, right_inliers)

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
