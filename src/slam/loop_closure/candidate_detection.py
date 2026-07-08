from __future__ import annotations

from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
from gtsam.symbol_shorthand import C

from src.slam.ba.gtsam_utils import (
    make_gtsam_stereo_calibration,
    pose3_from_world_to_camera_extrinsic,
)
from src.slam.pose_graph.constraints import symmetrize
from src.slam.pose_graph.covariance_routing import (
    CovarianceGraph,
    mahalanobis_squared,
)


@dataclass(frozen=True)
class LoopClosureCandidate:
    """A nonlocal keyframe pair ranked by relative-pose Mahalanobis score."""

    source_frame: int
    target_frame: int
    relative_pose: gtsam.Pose3
    relative_delta: np.ndarray
    covariance: np.ndarray
    mahalanobis_squared: float
    path_frame_ids: list[int]


def t2v(relative_pose: gtsam.Pose3) -> np.ndarray:
    """Return the 6D local coordinates of a relative Pose3 around identity.

    Ordering is GTSAM's Pose3 tangent convention:
        [rx, ry, rz, tx, ty, tz].
    """
    return np.asarray(
        gtsam.Pose3.Identity().localCoordinates(relative_pose),
        dtype=float,
    ).reshape(6)


def score_candidates_for_keyframe(
    *,
    optimized_values: gtsam.Values,
    covariance_graph: CovarianceGraph,
    keyframe_ids: list[int],
    target_frame: int,
    min_keyframe_separation: int = 5,
) -> list[LoopClosureCandidate]:
    """Score every temporally nonlocal earlier keyframe.

    No threshold is applied here. This is useful for empirical calibration
    of the Mahalanobis gate.
    """
    if len(keyframe_ids) != len(set(keyframe_ids)):
        raise ValueError("keyframe_ids must not contain duplicates.")

    if target_frame not in keyframe_ids:
        raise ValueError(f"Target frame {target_frame} is not a pose-graph keyframe.")

    if min_keyframe_separation < 1:
        raise ValueError("min_keyframe_separation must be at least 1.")

    target_index = keyframe_ids.index(target_frame)
    last_source_index = target_index - min_keyframe_separation

    if last_source_index < 0:
        return []

    target_key = C(target_frame)
    if not optimized_values.exists(target_key):
        raise ValueError(f"Missing optimized pose for target frame {target_frame}.")

    target_pose = optimized_values.atPose3(target_key)
    candidates: list[LoopClosureCandidate] = []

    for source_frame in keyframe_ids[: last_source_index + 1]:
        source_key = C(source_frame)

        if not optimized_values.exists(source_key):
            raise ValueError(f"Missing optimized pose for source frame {source_frame}.")

        covariance_path = covariance_graph.shortest_path(
            source_frame=source_frame,
            target_frame=target_frame,
        )
        if covariance_path is None:
            continue

        source_pose = optimized_values.atPose3(source_key)

        # Relative pose c_i^{-1} c_n.
        relative_pose = source_pose.between(target_pose)
        relative_delta = t2v(relative_pose)

        score = mahalanobis_squared(
            delta=relative_delta,
            covariance=covariance_path.covariance,
        )

        candidates.append(
            LoopClosureCandidate(
                source_frame=source_frame,
                target_frame=target_frame,
                relative_pose=relative_pose,
                relative_delta=relative_delta,
                covariance=covariance_path.covariance,
                mahalanobis_squared=score,
                path_frame_ids=covariance_path.frame_ids,
            )
        )

    candidates.sort(key=lambda candidate: candidate.mahalanobis_squared)
    return candidates


def detect_candidates_for_keyframe(
    *,
    optimized_values: gtsam.Values,
    covariance_graph: CovarianceGraph,
    keyframe_ids: list[int],
    target_frame: int,
    min_keyframe_separation: int = 5,
    mahalanobis_threshold: float,
    max_candidates: int | None = None,
) -> list[LoopClosureCandidate]:
    """Return candidates whose empirical Mahalanobis score passes the gate."""
    if mahalanobis_threshold <= 0:
        raise ValueError("mahalanobis_threshold must be positive.")

    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive or None.")

    scored_candidates = score_candidates_for_keyframe(
        optimized_values=optimized_values,
        covariance_graph=covariance_graph,
        keyframe_ids=keyframe_ids,
        target_frame=target_frame,
        min_keyframe_separation=min_keyframe_separation,
    )

    accepted = [
        candidate
        for candidate in scored_candidates
        if candidate.mahalanobis_squared <= mahalanobis_threshold
    ]

    if max_candidates is not None:
        accepted = accepted[:max_candidates]

    return accepted


def suppress_nearby_candidate_pairs(
    candidates: list[LoopClosureCandidate],
    *,
    keyframe_ids: list[int],
    source_radius: int = 2,
    target_radius: int = 2,
) -> list[LoopClosureCandidate]:
    """Keep the lowest-Mahalanobis representative from each local region.

    source_radius and target_radius are measured in keyframe-list positions,
    not raw frame numbers.
    """
    keyframe_index = {frame_id: index for index, frame_id in enumerate(keyframe_ids)}

    selected: list[LoopClosureCandidate] = []

    for candidate in sorted(
        candidates,
        key=lambda item: item.mahalanobis_squared,
    ):
        source_index = keyframe_index[candidate.source_frame]
        target_index = keyframe_index[candidate.target_frame]

        overlaps_existing = any(
            abs(source_index - keyframe_index[chosen.source_frame]) <= source_radius
            and abs(target_index - keyframe_index[chosen.target_frame]) <= target_radius
            for chosen in selected
        )

        if not overlaps_existing:
            selected.append(candidate)

    return selected


from src.slam.features.detectors import DEFAULT_ORB_NUM_FEATURES, FeatureType
from src.slam.geometry.correspondences import find_common_points
from src.slam.geometry.pnp import solve_pnp_safe
from src.slam.geometry.ransac import ransac_pnp
from src.slam.io.calibration import read_stereo_calibration
from src.slam.pipeline.stereo_pipeline import (
    StereoPointCloud,
    create_stereo_point_cloud,
)
from src.slam.pipeline.temporal_pipeline import match_left_frames

MIN_LOOP_FOUR_VIEW_MATCHES = 20
MIN_LOOP_INLIERS = 20


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
        *,
        feature_type: FeatureType = "akaze",
        min_loop_inliers: int = MIN_LOOP_INLIERS,
    ) -> None:
        if min_loop_inliers < 4:
            raise ValueError("min_loop_inliers must be at least 4.")

        self.feature_type = feature_type
        self.min_loop_inliers = min_loop_inliers
        self._stereo_cache: dict[int, StereoPointCloud] = {}

    def _get_stereo_cloud(self, frame_id: int) -> StereoPointCloud:
        """Load or compute the stereo point cloud of one frame."""
        if frame_id not in self._stereo_cache:
            self._stereo_cache[frame_id] = create_stereo_point_cloud(
                frame_id,
                reject_negative_depth=True,
                feature_type=self.feature_type,
                num_features=None,
                use_ratio_test=True,
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
            feature_type=self.feature_type,
            num_features=DEFAULT_ORB_NUM_FEATURES,
            use_ratio_test=True,
        )

        num_temporal_matches = len(temporal_data.matches)

        correspondences = find_common_points(
            source_cloud,
            target_cloud,
            temporal_data.left0_pts,
            temporal_data.left1_pts,
        )

        points_3d = correspondences.points_3d
        left0 = correspondences.left0
        right0 = correspondences.right0
        left1 = correspondences.left1
        right1 = correspondences.right1

        num_four_view_matches = len(points_3d)

        if num_four_view_matches < MIN_LOOP_FOUR_VIEW_MATCHES:
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
                    f"{num_four_view_matches} < {MIN_LOOP_FOUR_VIEW_MATCHES}"
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

        if num_inliers < self.min_loop_inliers:
            return ConsensusMatchResult(
                candidate=candidate,
                num_temporal_matches=num_temporal_matches,
                num_four_view_matches=num_four_view_matches,
                num_inliers=num_inliers,
                inlier_ratio=inlier_ratio,
                source_to_target_transform=transform_ransac,
                inlier_mask=inlier_mask,
                success=False,
                failure_reason=(f"Too few RANSAC inliers: {num_inliers} < {self.min_loop_inliers}"),
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


STEREO_PIXEL_SIGMA = 1.0
SOURCE_POSE_PRIOR_SIGMA = 1e-9


@dataclass(frozen=True)
class RelativePoseEstimate:
    """Relative pose and covariance refined by two-frame stereo BA."""

    consensus_result: ConsensusMatchResult
    relative_pose: gtsam.Pose3
    covariance: np.ndarray
    initial_error: float
    final_error: float
    num_landmarks: int


def _stereo_measurement(
    left_point: np.ndarray,
    right_point: np.ndarray,
) -> gtsam.StereoPoint2:
    """Convert rectified left/right image coordinates to GTSAM's stereo format:
    (u_left, u_right, v).
    """
    return gtsam.StereoPoint2(
        float(left_point[0]),
        float(right_point[0]),
        float(left_point[1]),
    )


def refine_relative_pose_with_bundle_adjustment(
    consensus_result: ConsensusMatchResult,
    *,
    pixel_sigma: float = STEREO_PIXEL_SIGMA,
) -> RelativePoseEstimate:
    """Refine one verified loop closure with a two-frame stereo bundle adjustment.

    The source camera is fixed as the local world frame. Therefore, the
    optimized target pose directly represents the source-to-target relative
    pose in camera-to-world Pose3 convention.
    """
    if not consensus_result.success:
        raise ValueError("Bundle refinement requires a successful consensus match.")

    if consensus_result.source_to_target_transform is None:
        raise ValueError("Missing PnP/RANSAC transform.")

    if consensus_result.inlier_mask is None:
        raise ValueError("Missing consensus inlier mask.")

    if any(
        array is None
        for array in (
            consensus_result.points_3d,
            consensus_result.source_left,
            consensus_result.source_right,
            consensus_result.target_left,
            consensus_result.target_right,
        )
    ):
        raise ValueError("Consensus result does not retain the four-view correspondences.")

    inlier_mask = np.asarray(
        consensus_result.inlier_mask,
        dtype=bool,
    ).reshape(-1)

    points_3d = np.asarray(consensus_result.points_3d, dtype=float)[inlier_mask]
    source_left = np.asarray(consensus_result.source_left, dtype=float)[inlier_mask]
    source_right = np.asarray(consensus_result.source_right, dtype=float)[inlier_mask]
    target_left = np.asarray(consensus_result.target_left, dtype=float)[inlier_mask]
    target_right = np.asarray(consensus_result.target_right, dtype=float)[inlier_mask]

    if len(points_3d) < 4:
        raise RuntimeError("Need at least four inliers for two-frame BA.")

    projection_left, projection_right = read_stereo_calibration()
    stereo_calibration = make_gtsam_stereo_calibration(
        projection_left,
        projection_right,
    )

    source_key = C(consensus_result.candidate.source_frame)
    target_key = C(consensus_result.candidate.target_frame)

    # The source camera defines the local world frame.
    source_pose = gtsam.Pose3()

    # PnP gives a source-camera -> target-camera world-to-camera transform.
    # Reuse the project utility that converts it to GTSAM's camera-to-world Pose3.
    target_pose_initial = pose3_from_world_to_camera_extrinsic(
        consensus_result.source_to_target_transform
    )

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    initial_values.insert(source_key, source_pose)
    initial_values.insert(target_key, target_pose_initial)

    source_prior_noise = gtsam.noiseModel.Isotropic.Sigma(
        6,
        SOURCE_POSE_PRIOR_SIGMA,
    )
    graph.add(
        gtsam.PriorFactorPose3(
            source_key,
            source_pose,
            source_prior_noise,
        )
    )

    stereo_noise = gtsam.noiseModel.Isotropic.Sigma(
        3,
        pixel_sigma,
    )

    for landmark_index, point_3d in enumerate(points_3d):
        landmark_key = gtsam.symbol("l", landmark_index)

        initial_values.insert(
            landmark_key,
            gtsam.Point3(
                float(point_3d[0]),
                float(point_3d[1]),
                float(point_3d[2]),
            ),
        )

        source_measurement = _stereo_measurement(
            source_left[landmark_index],
            source_right[landmark_index],
        )
        target_measurement = _stereo_measurement(
            target_left[landmark_index],
            target_right[landmark_index],
        )

        graph.add(
            gtsam.GenericStereoFactor3D(
                source_measurement,
                stereo_noise,
                source_key,
                landmark_key,
                stereo_calibration,
            )
        )
        graph.add(
            gtsam.GenericStereoFactor3D(
                target_measurement,
                stereo_noise,
                target_key,
                landmark_key,
                stereo_calibration,
            )
        )

    initial_error = graph.error(initial_values)

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_values,
    )
    optimized_values = optimizer.optimize()

    final_error = graph.error(optimized_values)

    optimized_source_pose = optimized_values.atPose3(source_key)
    optimized_target_pose = optimized_values.atPose3(target_key)

    relative_pose = optimized_source_pose.between(optimized_target_pose)

    marginals = gtsam.Marginals(
        graph,
        optimized_values,
    )

    # The source pose is effectively fixed, so target covariance is the
    # relative-pose covariance. This is the correct covariance for the
    # later BetweenFactorPose3 loop constraint.
    covariance = symmetrize(marginals.marginalCovariance(target_key))

    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues) <= 0:
        raise RuntimeError(
            "Two-frame BA produced a non-positive relative-pose covariance: "
            f"minimum eigenvalue = {np.min(eigenvalues):.3e}"
        )

    return RelativePoseEstimate(
        consensus_result=consensus_result,
        relative_pose=relative_pose,
        covariance=covariance,
        initial_error=initial_error,
        final_error=final_error,
        num_landmarks=len(points_3d),
    )


def select_spread_loop_closures(
    verified_results: list[ConsensusMatchResult],
    *,
    num_representatives: int = 3,
) -> list[ConsensusMatchResult]:
    """Select temporally spread representatives from one loop-overlap band.

    This avoids adding many highly correlated factors from nearly consecutive
    keyframes. For three representatives, this returns one near the start,
    one near the middle, and one near the end of the matched overlap.
    """
    successful = sorted(
        (result for result in verified_results if result.success),
        key=lambda result: result.candidate.target_frame,
    )

    if not successful:
        return []

    if num_representatives < 1:
        raise ValueError("num_representatives must be positive.")

    if len(successful) <= num_representatives:
        return successful

    selected_indices = np.linspace(
        0,
        len(successful) - 1,
        num=num_representatives,
        dtype=int,
    )

    return [successful[index] for index in sorted(set(selected_indices))]
