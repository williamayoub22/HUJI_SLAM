from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from ..io.calibration import read_calib
from ..geometry.triangulation import triangulate_opencv
from ..geometry.ransac import ransac_pnp
from ..features.detectors import extract_features, FeatureType
from ..features.matching import match_and_filter
from ..io.image_loader import read_images
from ..pipeline.stereo_pipeline import DEVIATION_THRESHOLD
from ..tracking_database import TrackingDB, Link

@dataclass
class DatabaseFrameData:
    """Container for the data needed to add one frame to the TrackingDB."""
    left_features: np.ndarray
    links: List[Link]
    points_3d: np.ndarray


def _links_to_points(links: List[Link]) -> Tuple[np.ndarray, np.ndarray]:
    """Converts frame links into aligned left and right image points."""
    left_pts = np.array([link.left_keypoint() for link in links], dtype=np.float64)
    right_pts = np.array([link.right_keypoint() for link in links], dtype=np.float64)
    return left_pts, right_pts


def _make_invalid_temporal_matches(
    num_prev_features: int,
) -> Tuple[List[cv2.DMatch], List[bool]]:
    """Creates invalid dummy matches, one for each previous-frame feature."""
    matches = [
        cv2.DMatch(_queryIdx=i, _trainIdx=0, _distance=float("inf"))
        for i in range(num_prev_features)
    ]
    inliers = [False] * num_prev_features
    return matches, inliers


def _empty_descriptor_array(descriptors: np.ndarray | None) -> np.ndarray:
    """Returns an empty descriptor array with a compatible descriptor dimension."""
    if descriptors is None:
        return np.empty((0, 0))

    return descriptors[:0]


def _compute_stereo_inlier_mask(
    kp_left: List[cv2.KeyPoint],
    kp_right: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    deviation_threshold: float,
) -> List[bool]:
    """Computes stereo inliers using the vertical-deviation criterion."""
    inliers = []

    for match in matches:
        y_left = kp_left[match.queryIdx].pt[1]
        y_right = kp_right[match.trainIdx].pt[1]
        inliers.append(abs(y_left - y_right) <= deviation_threshold)

    return inliers


def _create_frame_data(
    frame_idx: int,
    feature_type: FeatureType,
    num_features: int,
    ratio_threshold: float,
    deviation_threshold: float,
    P1: np.ndarray,
    P2: np.ndarray,
) -> DatabaseFrameData:
    """
    Extracts stereo features for one frame and converts valid stereo matches
    into TrackingDB-compatible descriptors, links, and aligned 3D points.
    """
    left_img, right_img = read_images(frame_idx)

    kp_left, des_left = extract_features(
        left_img,
        feature_type=feature_type,
        num_features=num_features,
    )
    kp_right, des_right = extract_features(
        right_img,
        feature_type=feature_type,
        num_features=num_features,
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
        feature_type=feature_type,
        ratio=ratio_threshold,
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
        deviation_threshold=deviation_threshold,
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
    feature_type: FeatureType,
    K: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    max_depth: float = 300.0,
) -> Tuple[List[cv2.DMatch], List[bool]]:
    """
    Matches descriptors from the previous left frame to the current left frame.

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

    if feature_type == "orb":
        norm_type = cv2.NORM_HAMMING
    else:
        norm_type = cv2.NORM_L2

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = matcher.match(prev_left_features, cur_left_features)

    match_by_query = {m.queryIdx: m for m in raw_matches}

    full_matches: List[cv2.DMatch] = []
    full_inliers: List[bool] = [False] * num_prev_features

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
        candidate_prev_3d,
        candidate_cur_left,
        candidate_prev_left,
        candidate_prev_right,
        candidate_cur_right,
        K,
        P1,
        P2,
    )

    if T_ransac is None or ransac_inlier_mask is None:
        return full_matches, full_inliers

    ransac_inlier_mask = np.asarray(ransac_inlier_mask, dtype=bool).flatten()

    for query_idx, is_inlier in zip(candidate_query_indices, ransac_inlier_mask):
        full_inliers[query_idx] = bool(is_inlier)

    return full_matches, full_inliers


def _count_frames(sequence_dir: Path) -> int:
    """Counts the number of frames in a KITTI-style sequence directory."""
    image_dir = sequence_dir / "image_0"
    return len(list(image_dir.glob("*.png")))


def load_tracking_database(base_filename: str | Path) -> TrackingDB:
    """
    Loads a TrackingDB saved with TrackingDB.serialize().

    Returns:
         db: the loaded tracking database.
    """
    db = TrackingDB()
    db.load(str(base_filename))
    return db

def build_tracking_database(
    sequence_dir: Path,
    num_frames: int | None = None,
    feature_type: FeatureType = "orb",
    num_features: int = 3000,
    ratio_threshold: float = 0.75,
    deviation_threshold: float = DEVIATION_THRESHOLD,
) -> Tuple[TrackingDB, List[float]]:
    """
    Builds a TrackingDB over a sequence of stereo frames.

    Returns:
        db: tracking database containing all tracks.
        inlier_percentages: percentage of RANSAC-PnP inliers per frame transition.
    """
    total_frames = _count_frames(sequence_dir)

    if num_frames is None:
        num_frames = total_frames
    else:
        num_frames = min(num_frames, total_frames)

    P1, P2 = read_calib()
    K = P1[:, :3]

    db = TrackingDB()
    inlier_percentages = []

    prev_frame_data = None

    for frame_idx in tqdm(range(num_frames), desc="Building tracking database"):
        cur_frame_data = _create_frame_data(
            frame_idx=frame_idx,
            feature_type=feature_type,
            num_features=num_features,
            ratio_threshold=ratio_threshold,
            deviation_threshold=deviation_threshold,
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
                feature_type=feature_type,
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
                inlier_percentages.append(
                    100.0 * sum(temporal_inliers) / len(temporal_inliers)
                )

        prev_frame_data = cur_frame_data

    return db, inlier_percentages
