"""Build and load the feature-tracking database for a stereo sequence."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import gtsam
import numpy as np
from tqdm import tqdm

from src.slam.data.db_facade import SlamDatabase
from src.slam.data.tracking_db import Link, TrackingDB
from src.slam.features.detectors import FeatureType, extract_features
from src.slam.features.matching import match_and_filter
from src.slam.geometry.ransac import ransac_pnp
from src.slam.geometry.transforms import to_homogeneous_transform
from src.slam.geometry.triangulation import triangulate_opencv
from src.slam.io.calibration import read_stereo_calibration
from src.slam.io.image_loader import read_images
from src.slam import config
from src.slam.geometry.stereo import make_gtsam_stereo_calibration
from src.slam.pipeline.frame_processing import process_stereo_frame, track_between_frames


@dataclass
class DatabaseFrameData:
    """Container for the data needed to add one frame to the TrackingDB."""

    left_features: np.ndarray
    links: list[Link]
    points_3d: np.ndarray


def _links_to_points(links: list[Link]) -> tuple[np.ndarray, np.ndarray]:
    """Convert frame links into aligned left and right image points."""
    left_pts = np.array([link.left_keypoint() for link in links], dtype=np.float64)
    right_pts = np.array([link.right_keypoint() for link in links], dtype=np.float64)
    return left_pts, right_pts


def _make_invalid_temporal_matches(
    num_prev_features: int,
) -> tuple[list[cv2.DMatch], list[bool]]:
    """Create invalid dummy matches, one for each previous-frame feature."""
    matches = [
        cv2.DMatch(_queryIdx=i, _trainIdx=0, _distance=float("inf"))
        for i in range(num_prev_features)
    ]
    inliers = [False] * num_prev_features
    return matches, inliers


def _empty_descriptor_array(descriptors: np.ndarray | None) -> np.ndarray:
    """Return an empty descriptor array with a compatible descriptor dimension."""
    if descriptors is None:
        return np.empty((0, 0))

    return descriptors[:0]


def _compute_stereo_inlier_mask(
    kp_left: list[cv2.KeyPoint],
    kp_right: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    deviation_threshold: float,
) -> list[bool]:
    """Compute stereo inliers using the vertical-deviation criterion."""
    inliers = []

    for match in matches:
        x_left = kp_left[match.queryIdx].pt[0]
        y_left = kp_left[match.queryIdx].pt[1]
        x_right = kp_right[match.trainIdx].pt[0]
        y_right = kp_right[match.trainIdx].pt[1]

        valid_y = abs(y_left - y_right) <= deviation_threshold
        valid_x = (x_left - x_right) > config.MIN_DISPARITY  # Positive disparity check

        inliers.append(valid_y and valid_x)

    return inliers


def _create_frame_data(
    frame_idx: int,
    P1: np.ndarray,
    P2: np.ndarray,
) -> DatabaseFrameData:
    """Extract stereo features for one frame.

    Convert valid stereo matches into TrackingDB-compatible descriptors, links,
    and aligned 3D points.
    """
    left_img, right_img = read_images(frame_idx)

    kp_left, des_left = extract_features(
        left_img,
        feature_type=config.FEATURE_TYPE,
        num_features=config.NUM_FEATURES,
    )
    kp_right, des_right = extract_features(
        right_img,
        feature_type=config.FEATURE_TYPE,
        num_features=config.NUM_FEATURES,
    )

    if des_left is None or des_right is None:
        empty_des = _empty_descriptor_array(des_left)
        return DatabaseFrameData(
            left_features=empty_des,
            links=[],
            points_3d=np.empty((0, 3)),
        )

    stereo_matches = match_and_filter(
        des_left,
        des_right,
        feature_type=config.FEATURE_TYPE,
        ratio=config.RATIO_THRESHOLD,
    )

    if len(stereo_matches) == 0:
        return DatabaseFrameData(
            left_features=_empty_descriptor_array(des_left),
            links=[],
            points_3d=np.empty((0, 3)),
        )

    stereo_inliers = _compute_stereo_inlier_mask(
        kp_left=kp_left,
        kp_right=kp_right,
        matches=stereo_matches,
        deviation_threshold=config.DEVIATION_THRESHOLD,
    )

    left_features, links = TrackingDB.create_links(
        features=des_left,
        kp_left=kp_left,
        kp_right=kp_right,
        matches=stereo_matches,
        inliers=stereo_inliers,
    )

    if len(links) == 0:
        return DatabaseFrameData(
            left_features=left_features,
            links=links,
            points_3d=np.empty((0, 3)),
        )

    left_pts, right_pts = _links_to_points(links)
    points_3d = triangulate_opencv(P1, P2, left_pts, right_pts)

    return DatabaseFrameData(
        left_features=left_features,
        links=links,
        points_3d=points_3d,
    )


def _match_temporal_features(
    prev_frame: DatabaseFrameData,
    cur_frame: DatabaseFrameData,
    K: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    max_depth: float = config.MAX_DEPTH,
) -> tuple[list[cv2.DMatch], list[bool]]:
    """Match descriptors from the previous left frame to the current left frame.

    The TrackingDB expects one temporal match entry per previous-frame feature.
    We therefore first compute a raw 1-NN match for each previous descriptor,
    and then use the Exercise 3 RANSAC-PnP consensus test to decide which
    matches are valid inliers.
    """
    prev_left_features = prev_frame.left_features
    cur_left_features = cur_frame.left_features

    if prev_left_features is None:
        return [], []

    num_prev_features = len(prev_left_features)

    if num_prev_features == 0:
        return [], []

    if cur_left_features is None or len(cur_left_features) == 0:
        return _make_invalid_temporal_matches(num_prev_features)

    norm_type = cv2.NORM_HAMMING if config.FEATURE_TYPE in ("orb", "akaze") else cv2.NORM_L2

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = matcher.match(prev_left_features, cur_left_features)

    match_by_query = {match.queryIdx: match for match in raw_matches}

    full_matches: list[cv2.DMatch] = []
    full_inliers: list[bool] = [False] * num_prev_features

    for query_idx in range(num_prev_features):
        if query_idx in match_by_query:
            full_matches.append(match_by_query[query_idx])
        else:
            full_matches.append(
                cv2.DMatch(
                    _queryIdx=query_idx,
                    _trainIdx=0,
                    _distance=float("inf"),
                )
            )

    prev_left_pts, prev_right_pts = _links_to_points(prev_frame.links)
    cur_left_pts, cur_right_pts = _links_to_points(cur_frame.links)

    candidate_prev_3d = []
    candidate_prev_left = []
    candidate_prev_right = []
    candidate_cur_left = []
    candidate_cur_right = []
    candidate_query_indices = []

    for match in full_matches:
        if not np.isfinite(match.distance):
            continue

        prev_idx = match.queryIdx
        cur_idx = match.trainIdx

        if prev_idx >= len(prev_frame.points_3d) or cur_idx >= len(cur_frame.links):
            continue

        point_3d = prev_frame.points_3d[prev_idx]

        if not np.all(np.isfinite(point_3d)):
            continue

        if point_3d[2] <= 0 or point_3d[2] >= max_depth:
            continue

        candidate_prev_3d.append(point_3d)
        candidate_prev_left.append(prev_left_pts[prev_idx])
        candidate_prev_right.append(prev_right_pts[prev_idx])
        candidate_cur_left.append(cur_left_pts[cur_idx])
        candidate_cur_right.append(cur_right_pts[cur_idx])
        candidate_query_indices.append(prev_idx)

    if len(candidate_prev_3d) < 4:
        return full_matches, full_inliers

    candidate_prev_3d = np.asarray(candidate_prev_3d)
    candidate_prev_left = np.asarray(candidate_prev_left)
    candidate_prev_right = np.asarray(candidate_prev_right)
    candidate_cur_left = np.asarray(candidate_cur_left)
    candidate_cur_right = np.asarray(candidate_cur_right)

    T_ransac, ransac_inlier_mask = ransac_pnp(
        points_3d=candidate_prev_3d,
        left1=candidate_cur_left,
        left0=candidate_prev_left,
        right0=candidate_prev_right,
        right1=candidate_cur_right,
        intrinsic_matrix=K,
        left_projection_matrix=P1,
        right_projection_matrix=P2,
    )

    if T_ransac is None or ransac_inlier_mask is None:
        return full_matches, full_inliers

    ransac_inlier_mask = np.asarray(ransac_inlier_mask, dtype=bool).flatten()

    for query_idx, is_inlier in zip(
        candidate_query_indices,
        ransac_inlier_mask,
        strict=True,
    ):
        full_inliers[query_idx] = bool(is_inlier)

    return full_matches, full_inliers


def _count_frames(sequence_dir: Path) -> int:
    """Count the number of frames in a KITTI-style sequence directory."""
    image_dir = sequence_dir / "image_0"
    return len(list(image_dir.glob("*.png")))


def load_tracking_database(base_filename: str | Path) -> TrackingDB:
    """Load a TrackingDB saved with ``TrackingDB.serialize()``.

    Returns:
        The loaded tracking database.
    """
    db = TrackingDB()
    db.load(str(base_filename))
    return db


def build_tracking_database(
    sequence_dir: Path,
    num_frames: int | None = None,
) -> tuple[TrackingDB, list[float]]:
    """Build a TrackingDB over a sequence of stereo frames.

    Returns:
        A tracking database and the RANSAC-PnP inlier percentage for each
        consecutive frame transition.
    """
    total_frames = _count_frames(sequence_dir)
    num_frames = total_frames if num_frames is None else min(num_frames, total_frames)

    P1, P2 = read_stereo_calibration()
    K = P1[:, :3]

    db = TrackingDB()
    inlier_percentages = []

    prev_frame_data = None

    for frame_idx in tqdm(range(num_frames), desc="Building tracking database"):
        cur_frame_data = _create_frame_data(
            frame_idx=frame_idx,
            P1=P1,
            P2=P2,
        )

        if frame_idx == 0:
            db.add_frame(
                links=cur_frame_data.links,
                left_features=cur_frame_data.left_features,
            )
        else:
            temporal_matches, temporal_inliers = _match_temporal_features(
                prev_frame=prev_frame_data,
                cur_frame=cur_frame_data,
                K=K,
                P1=P1,
                P2=P2,
            )

            db.add_frame(
                links=cur_frame_data.links,
                left_features=cur_frame_data.left_features,
                matches_to_previous_left=temporal_matches,
                inliers=temporal_inliers,
            )

            if len(temporal_inliers) == 0:
                inlier_percentages.append(0.0)
            else:
                inlier_percentages.append(100.0 * sum(temporal_inliers) / len(temporal_inliers))

        prev_frame_data = cur_frame_data

    return db, inlier_percentages


def build_database():
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    P1, P2 = read_stereo_calibration()
    K = P1[:, :3]
    calibration = make_gtsam_stereo_calibration(P1, P2)

    db_path = config.CACHE_DIR / 'slam_db'
    full_db_path = config.CACHE_DIR / 'slam_db.pkl'

    if full_db_path.exists():
        print(f'Loading database from {full_db_path}...')
        slam_db = SlamDatabase.load(str(db_path))
        return slam_db, calibration

    print('Building database from scratch...')
    slam_db = SlamDatabase()

    total_frames = len(list((config.SEQUENCE_DIR / "image_0").glob("*.png")))
    global_T = np.eye(4)
    slam_db.manager_poses.add_pose(0, global_T)
    global_camera_matrices = [global_T[:3, :].copy()]
    inlier_percentages = []

    prev_frame_data = process_stereo_frame(0, P1, P2)
    slam_db.manager_2d.add_frame(
        links=prev_frame_data.links,
        left_features=prev_frame_data.left_features
    )

    for frame_idx in tqdm(range(1, total_frames), desc='Building SLAM Database'):
        cur_frame_data = process_stereo_frame(frame_idx, P1, P2)
        
        tracking_result = track_between_frames(
            prev_frame_data, cur_frame_data, K, P1, P2
        )
        
        inlier_percentages.append(tracking_result.inlier_ratio)

        step_T = to_homogeneous_transform(tracking_result.relative_pose)
        global_T = step_T @ global_T
        slam_db.manager_poses.add_pose(frame_idx, global_T)
        global_camera_matrices.append(global_T[:3, :].copy())

        slam_db.manager_2d.add_frame(
            links=cur_frame_data.links,
            left_features=cur_frame_data.left_features,
            matches_to_previous_left=tracking_result.matches,
            inliers=tracking_result.inliers
        )
        
        prev_frame_data = cur_frame_data

    slam_db.triangulate_all_tracks(P1, P2)
    
    slam_db.serialize(str(db_path))
    np.save(config.CACHE_DIR / 'global_camera_matrices.npy', np.array(global_camera_matrices, dtype=float))
    np.save(config.CACHE_DIR / 'inlier_percentages.npy', np.array(inlier_percentages, dtype=float))

    return slam_db, calibration