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
) -> list[LoopClosureCandidate]:
    """Score every temporally nonlocal earlier keyframe.

    No threshold is applied here. This is useful for empirical calibration
    of the Mahalanobis gate.
    """
    if len(keyframe_ids) != len(set(keyframe_ids)):
        raise ValueError("keyframe_ids must not contain duplicates.")

    if target_frame not in keyframe_ids:
        raise ValueError(f"Target frame {target_frame} is not a pose-graph keyframe.")

    if config.MIN_KEYFRAME_SEPARATION < 1:
        raise ValueError("min_keyframe_separation must be at least 1.")

    target_index = keyframe_ids.index(target_frame)
    last_source_index = target_index - config.MIN_KEYFRAME_SEPARATION

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


from src.slam.geometry.correspondences import find_common_points
from src.slam.geometry.projection import solve_pnp_safe
from src.slam.geometry.ransac import ransac_pnp
from src.slam.io.calibration import read_stereo_calibration
from src.slam.pipeline.stereo_pipeline import (
    StereoPointCloud,
    create_stereo_point_cloud,
)
from src.slam.pipeline.temporal_pipeline import match_left_frames

from src.slam.pipeline.temporal_pipeline import match_left_frames



def select_spread_loop_closures(
    verified_results: list[ConsensusMatchResult],
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

    if config.NUM_LOOP_CLOSURE_REPRESENTATIVES < 1:
        raise ValueError("num_representatives must be positive.")

    if len(successful) <= config.NUM_LOOP_CLOSURE_REPRESENTATIVES:
        return successful

    selected_indices = np.linspace(
        0,
        len(successful) - 1,
        num=config.NUM_LOOP_CLOSURE_REPRESENTATIVES,
        dtype=int,
    )

    return [successful[index] for index in sorted(set(selected_indices))]
