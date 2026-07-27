"""
Provides candidates components and utilities for the SLAM pipeline.
"""

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
    # Extract the 6D tangent vector from the identity pose to the given relative_pose
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
    # Ensure all keyframes are unique to prevent indexing collisions
    if len(keyframe_ids) != len(set(keyframe_ids)):
        raise ValueError("keyframe_ids must not contain duplicates.")

    # Validate that the requested target frame is actively tracked in our pose graph
    if target_frame not in keyframe_ids:
        raise ValueError(f"Target frame {target_frame} is not a pose-graph keyframe.")

    # Guard against invalid configuration values for keyframe separation
    if config.MIN_KEYFRAME_SEPARATION < 1:
        raise ValueError("min_keyframe_separation must be at least 1.")

    # Calculate the boundary index to prevent matching with immediate temporal neighbors
    target_index = keyframe_ids.index(target_frame)
    last_source_index = target_index - config.MIN_KEYFRAME_SEPARATION

    # If there aren't enough historical frames to satisfy the separation, return early
    if last_source_index < 0:
        return []

    # Retrieve the optimized pose for the target frame from GTSAM values
    target_key = C(target_frame)
    if not optimized_values.exists(target_key):
        raise ValueError(f"Missing optimized pose for target frame {target_frame}.")

    target_pose = optimized_values.atPose3(target_key)
    candidates: list[LoopClosureCandidate] = []

    # Iterate through all valid historical source frames up to the separation boundary
    for source_frame in keyframe_ids[: last_source_index + 1]:
        source_key = C(source_frame)

        if not optimized_values.exists(source_key):
            raise ValueError(f"Missing optimized pose for source frame {source_frame}.")

        # Find the shortest path in the covariance graph to get the relative uncertainty
        covariance_path = covariance_graph.shortest_path(
            source_frame=source_frame,
            target_frame=target_frame,
        )
        # Skip if no valid path exists between these frames in the covariance graph
        if covariance_path is None:
            continue

        source_pose = optimized_values.atPose3(source_key)

        # Compute the relative pose (c_i^{-1} c_n) and its 6D local coordinate vector
        relative_pose = source_pose.between(target_pose)
        relative_delta = t2v(relative_pose)

        # Calculate the Mahalanobis distance to measure how likely this loop closure is
        score = mahalanobis_squared(
            delta=relative_delta,
            covariance=covariance_path.covariance,
        )

        # Store the computed candidate metrics
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

    # Sort all valid candidates from lowest Mahalanobis distance (most likely) to highest
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
    # Sanity checks for gating parameters
    if mahalanobis_threshold <= 0:
        raise ValueError("mahalanobis_threshold must be positive.")

    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive or None.")

    # Retrieve all scored candidates for the target frame without thresholding first
    scored_candidates = score_candidates_for_keyframe(
        optimized_values=optimized_values,
        covariance_graph=covariance_graph,
        keyframe_ids=keyframe_ids,
        target_frame=target_frame,
    )

    # Filter out any candidates that exceed our statistical distance threshold
    accepted = [
        candidate
        for candidate in scored_candidates
        if candidate.mahalanobis_squared <= mahalanobis_threshold
    ]

    # Optionally truncate the list to the top N candidates (already sorted by score)
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
    # Create a lookup mapping from frame IDs to their sequential index in the keyframe list
    keyframe_index = {frame_id: index for index, frame_id in enumerate(keyframe_ids)}

    selected: list[LoopClosureCandidate] = []

    # Iterate through candidates, prioritizing those with the best (lowest) Mahalanobis scores
    for candidate in sorted(
        candidates,
        key=lambda item: item.mahalanobis_squared,
    ):
        source_index = keyframe_index[candidate.source_frame]
        target_index = keyframe_index[candidate.target_frame]

        # Check if the current candidate falls within the exclusion radius of any already-selected candidate
        overlaps_existing = any(
            abs(source_index - keyframe_index[chosen.source_frame]) <= source_radius
            and abs(target_index - keyframe_index[chosen.target_frame]) <= target_radius
            for chosen in selected
        )

        # Only add the candidate if it represents a new, distinct region of overlap
        if not overlaps_existing:
            selected.append(candidate)

    return selected


# Original mid-file imports preserved exactly
from src.slam.geometry.correspondences import find_common_points
from src.slam.geometry.projection import solve_pnp_safe
from src.slam.geometry.ransac import ransac_pnp
from src.slam.io.calibration import read_stereo_calibration
from src.slam.pipeline.temporal_pipeline import match_left_frames


def select_spread_loop_closures(
    verified_results: list[ConsensusMatchResult],
) -> list[ConsensusMatchResult]:
    """Select temporally spread representatives from one loop-overlap band.

    This avoids adding many highly correlated factors from nearly consecutive
    keyframes. For three representatives, this returns one near the start,
    one near the middle, and one near the end of the matched overlap.
    """
    # Filter for matches that succeeded and sort them chronologically by target frame
    successful = sorted(
        (result for result in verified_results if result.success),
        key=lambda result: result.candidate.target_frame,
    )

    # Return empty list if no matches were successful
    if not successful:
        return []

    # Validate that we are requesting at least 1 representative match
    if config.NUM_LOOP_CLOSURE_REPRESENTATIVES < 1:
        raise ValueError("num_representatives must be positive.")

    # If the number of successful matches is less than or equal to our requested target, keep them all
    if len(successful) <= config.NUM_LOOP_CLOSURE_REPRESENTATIVES:
        return successful

    # Use linspace to calculate evenly distributed index positions across the successful matches
    selected_indices = np.linspace(
        0,
        len(successful) - 1,
        num=config.NUM_LOOP_CLOSURE_REPRESENTATIVES,
        dtype=int,
    )

    # Extract the evenly spaced representatives, ensuring uniqueness by converting to a set then sorting
    return [successful[index] for index in sorted(set(selected_indices))]
