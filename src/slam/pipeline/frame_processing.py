from dataclasses import dataclass

import cv2
import numpy as np

from src.slam import config
from src.slam.data.tracking_db import Link, TrackingDB
from src.slam.pipeline.stereo_pipeline import create_stereo_cloud
from src.slam.geometry.ransac import ransac_pnp


@dataclass
class FrameData:
    """Contains features, intra-frame stereo links, and 3D points for one frame."""

    left_features: np.ndarray | None
    links: list[Link]
    points_3d: np.ndarray


@dataclass
class TrackingResult:
    """Result of tracking between two frames."""

    matches: list[cv2.DMatch]
    inliers: list[bool]
    inlier_ratio: float
    relative_pose: np.ndarray


def links_to_points(links: list[Link]) -> tuple[np.ndarray, np.ndarray]:
    """Convert TrackingDB Links back to aligned (N, 2) pixel coordinates."""
    left_pts = np.asarray([link.left_keypoint() for link in links], dtype=float)
    right_pts = np.asarray([link.right_keypoint() for link in links], dtype=float)

    if len(left_pts) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    return left_pts, right_pts


def _empty_descriptor_array(prototype: np.ndarray | None) -> np.ndarray:
    """Create a valid empty descriptor array."""
    if prototype is None:
        return np.empty((0, 128), dtype=np.float32)

    return np.empty((0, prototype.shape[1]), dtype=prototype.dtype)


def _make_invalid_temporal_matches(num_queries: int) -> tuple[list[cv2.DMatch], list[bool]]:
    """Return all-invalid matches when the target frame has no features."""
    matches = [
        cv2.DMatch(_queryIdx=i, _trainIdx=0, _distance=float("inf"))
        for i in range(num_queries)
    ]
    inliers = [False] * num_queries
    return matches, inliers


def process_stereo_frame(
    frame_idx: int,
    P1: np.ndarray,
    P2: np.ndarray,
) -> FrameData:
    """Extract stereo features for one frame using create_stereo_cloud.
    
    Converts valid stereo matches into TrackingDB-compatible descriptors, links,
    and aligned 3D points.
    """
    cloud = create_stereo_cloud(
        frame_idx,
        reject_negative_depth=False,
        ratio_threshold=config.RATIO_THRESHOLD,
    )
    
    data = cloud.data

    if len(data.matches) == 0:
        return FrameData(
            left_features=_empty_descriptor_array(data.left_descriptors),
            links=[],
            points_3d=np.empty((0, 3)),
        )

    from src.slam.pipeline.stereo_pipeline import get_stereo_inlier_mask
    stereo_inliers = get_stereo_inlier_mask(data)

    left_features, links = TrackingDB.create_links(
        features=data.left_descriptors,
        kp_left=data.kp_left,
        kp_right=data.kp_right,
        matches=data.matches,
        inliers=stereo_inliers,
    )

    if len(links) == 0:
        return FrameData(
            left_features=left_features,
            links=links,
            points_3d=np.empty((0, 3)),
        )

    return FrameData(
        left_features=left_features,
        links=links,
        points_3d=cloud.points_3d,
    )


def match_temporal_features(
    prev_frame: FrameData,
    cur_frame: FrameData,
    K: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    max_depth: float = config.MAX_DEPTH,
) -> tuple[list[cv2.DMatch], list[bool]]:
    """Match descriptors from the previous left frame to the current left frame."""
    prev_left_features = prev_frame.left_features
    cur_left_features = cur_frame.left_features

    if prev_left_features is None or len(prev_left_features) == 0:
        return [], []

    num_prev_features = len(prev_left_features)

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

    prev_left_pts, prev_right_pts = links_to_points(prev_frame.links)
    cur_left_pts, cur_right_pts = links_to_points(cur_frame.links)

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


def estimate_relative_pose(
    prev_frame: FrameData,
    cur_frame: FrameData,
    temporal_matches: list[cv2.DMatch],
    K: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """Estimate relative pose from temporal matches using PnP/RANSAC."""
    candidate_prev_3d = []
    candidate_prev_left = []
    candidate_prev_right = []
    candidate_cur_left = []
    candidate_cur_right = []

    for match in temporal_matches:
        if not np.isfinite(match.distance):
            continue
        prev_idx = match.queryIdx
        cur_idx = match.trainIdx
        
        if prev_idx < len(prev_frame.points_3d) and cur_idx < len(cur_frame.links):
            p3d = prev_frame.points_3d[prev_idx]
            if np.all(np.isfinite(p3d)) and 0 < p3d[2] < config.MAX_DEPTH:
                candidate_prev_3d.append(p3d)
                candidate_prev_left.append(prev_frame.links[prev_idx].left_keypoint())
                candidate_prev_right.append(prev_frame.links[prev_idx].right_keypoint())
                candidate_cur_left.append(cur_frame.links[cur_idx].left_keypoint())
                candidate_cur_right.append(cur_frame.links[cur_idx].right_keypoint())

    T_rel = np.eye(4)
    if len(candidate_prev_3d) >= 4:
        T_candidate, _ = ransac_pnp(
            points_3d=np.asarray(candidate_prev_3d),
            left1=np.asarray(candidate_cur_left),
            left0=np.asarray(candidate_prev_left),
            right0=np.asarray(candidate_prev_right),
            right1=np.asarray(candidate_cur_right),
            intrinsic_matrix=K,
            left_projection_matrix=P1,
            right_projection_matrix=P2,
        )
        if T_candidate is not None:
            T_rel = T_candidate
            
    return T_rel


def track_between_frames(
    prev_frame: FrameData,
    cur_frame: FrameData,
    K: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> TrackingResult:
    """Full frame-to-frame tracking: temporal match -> RANSAC -> relative pose."""
    matches, inliers = match_temporal_features(prev_frame, cur_frame, K, P1, P2)
    inlier_ratio = sum(inliers) / len(matches) if len(matches) > 0 else 0.0
    relative_pose = estimate_relative_pose(prev_frame, cur_frame, matches, K, P1, P2)

    return TrackingResult(
        matches=matches,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        relative_pose=relative_pose,
    )
