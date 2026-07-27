from __future__ import annotations

from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
from gtsam.symbol_shorthand import C

from src.slam.geometry.stereo import (
    make_gtsam_stereo_calibration,
    pose3_from_world_to_camera_extrinsic,
)
from src.slam.pose_graph.constraints import symmetrize
from src.slam.pose_graph.covariance_routing import (
    CovarianceGraph,
    mahalanobis_squared,
)
from .. import config
from src.slam.pipeline.stereo_pipeline import create_stereo_point_cloud, StereoPointCloud
from src.slam.pipeline.temporal_pipeline import match_left_frames
from src.slam.geometry.correspondences import find_common_points
from src.slam.io.calibration import read_stereo_calibration
from src.slam.geometry.ransac import ransac_pnp
from src.slam.geometry.projection import solve_pnp_safe
from src.slam.loop_closure.candidates import LoopClosureCandidate


@dataclass(frozen=True)
class ConsensusMatchResult:
    """Geometric verification result for one loop-closure candidate."""

    candidate: LoopClosureCandidate
    num_temporal_matches: int
    num_four_view_matches: int
    num_inliers: int
    inlier_ratio: float
    source_to_target_transform: np.ndarray | None
    inlier_mask: np.ndarray | None
    success: bool
    failure_reason: str | None = None

    # Kept for the two-frame BA stage.
    points_3d: np.ndarray | None = None
    source_left: np.ndarray | None = None
    source_right: np.ndarray | None = None
    target_left: np.ndarray | None = None
    target_right: np.ndarray | None = None


class ConsensusMatcher:
    """Reuses the Exercise-3 stereo + temporal matching + PnP/RANSAC pipeline
    for wide-baseline loop-closure verification.

    Stereo point clouds are cached because one keyframe can occur in more
    than one candidate pair.
    """

    def __init__(
        self,
    ) -> None:
        if config.MIN_LOOP_INLIERS < 4:
            raise ValueError("min_loop_inliers must be at least 4.")

        self._stereo_cache: dict[int, StereoPointCloud] = {}

    def _get_stereo_cloud(self, frame_id: int) -> StereoPointCloud:
        """Load or compute the stereo point cloud of one frame."""
        if frame_id not in self._stereo_cache:
            self._stereo_cache[frame_id] = create_stereo_point_cloud(
                frame_id,
                reject_negative_depth=True,
                use_custom_triangulation=False,
            )

        return self._stereo_cache[frame_id]

    def verify(
        self,
        candidate: LoopClosureCandidate,
    ) -> ConsensusMatchResult:
        """Verify one candidate with the Exercise-3 four-view consensus pipeline.

        The returned transformation maps 3D points represented in the source
        left-camera coordinates to the target left-camera coordinates.
        """
        source_frame = candidate.source_frame
        target_frame = candidate.target_frame

        source_cloud = self._get_stereo_cloud(source_frame)
        target_cloud = self._get_stereo_cloud(target_frame)

        temporal_data = match_left_frames(
            source_frame,
            target_frame,
        )

        num_temporal_matches = len(temporal_data.matches)

        correspondences = find_common_points(
            source_cloud.points_3d,
            source_cloud.left_inliers,
            source_cloud.right_inliers,
            target_cloud.left_inliers,
            target_cloud.right_inliers,
            temporal_data.left0_pts,
            temporal_data.left1_pts,
        )

        points_3d = correspondences.points_3d
        left0 = correspondences.left0
        right0 = correspondences.right0
        left1 = correspondences.left1
        right1 = correspondences.right1

        num_four_view_matches = len(points_3d)

        if num_four_view_matches < config.MIN_LOOP_INLIERS:
            return ConsensusMatchResult(
                candidate=candidate,
                num_temporal_matches=num_temporal_matches,
                num_four_view_matches=num_four_view_matches,
                num_inliers=0,
                inlier_ratio=0.0,
                source_to_target_transform=None,
                inlier_mask=None,
                success=False,
                failure_reason=(
                    "Too few four-view correspondences: "
                    f"{num_four_view_matches} < {config.MIN_LOOP_INLIERS}"
                ),
            )

        projection_left, projection_right = read_stereo_calibration()
        calibration_matrix = projection_left[:, :3]

        transform_ransac, inlier_mask = ransac_pnp(
            points_3d,
            left1,
            left0,
            right0,
            right1,
            calibration_matrix,
            projection_left,
            projection_right,
        )

        if transform_ransac is None or inlier_mask is None:
            return ConsensusMatchResult(
                candidate=candidate,
                num_temporal_matches=num_temporal_matches,
                num_four_view_matches=num_four_view_matches,
                num_inliers=0,
                inlier_ratio=0.0,
                source_to_target_transform=None,
                inlier_mask=None,
                success=False,
                failure_reason="PnP/RANSAC did not find a valid transformation.",
            )

        inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        num_inliers = int(np.sum(inlier_mask))
        inlier_ratio = num_inliers / num_four_view_matches

        if num_inliers < config.MIN_LOOP_INLIERS:
            return ConsensusMatchResult(
                candidate=candidate,
                num_temporal_matches=num_temporal_matches,
                num_four_view_matches=num_four_view_matches,
                num_inliers=num_inliers,
                inlier_ratio=inlier_ratio,
                source_to_target_transform=transform_ransac,
                inlier_mask=inlier_mask,
                success=False,
                failure_reason=(f"Too few RANSAC inliers: {num_inliers} < {config.MIN_LOOP_INLIERS}"),
            )

        transform_refined = solve_pnp_safe(
            points_3d[inlier_mask],
            left1[inlier_mask],
            calibration_matrix,
            cv2.SOLVEPNP_EPNP,
        )

        if transform_refined is None:
            transform_refined = transform_ransac

        return ConsensusMatchResult(
            candidate=candidate,
            num_temporal_matches=num_temporal_matches,
            num_four_view_matches=num_four_view_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            source_to_target_transform=transform_refined,
            inlier_mask=inlier_mask,
            success=True,
            points_3d=points_3d,
            source_left=left0,
            source_right=right0,
            target_left=left1,
            target_right=right1,
        )


SOURCE_POSE_PRIOR_SIGMA = 1e-9
