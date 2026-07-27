"""
Pose Graph Optimization Pipeline for SLAM.

This module is responsible for taking the local Bundle Adjustment solutions and integrating them 
into a unified global Pose Graph. It handles the detection of loop closures and the subsequent 
optimization of the pose graph to correct accumulated drift.

Key features:
- Transforms local BA relative motions into GTSAM `BetweenFactorPose3` constraints.
- Detects temporally non-local loop closures using a Mahalanobis distance gate.
- Creates `pose_graph_no_lc` (pure VO) and `pose_graph_with_lc` (global drift corrected) models.
"""

import gtsam

# solve BA windows and extract all constraints
import numpy as np
from gtsam.symbol_shorthand import C

from src.slam import config
from src.slam.bundle_adjustment.window_selection import choose_keyframes_by_motion
from src.slam.bundle_adjustment.window_solver import solve_all_bundle_windows
from src.slam.data.db_facade import SlamDatabase
from src.slam.geometry.stereo import (
    make_gtsam_stereo_calibration,
    pose3_from_world_to_camera_extrinsic,
)
from src.slam.loop_closure.candidates import (
    LoopClosureCandidate,
    detect_candidates_for_keyframe,
    score_candidates_for_keyframe,
)
from src.slam.loop_closure.diagnostics import (
    _print_mahalanobis_threshold_sweep,
    _show_candidate_pair,
)
from src.slam.loop_closure.refinement import RelativePoseEstimate
from src.slam.pipeline.database_pipeline import build_database
from src.slam.pose_graph.constraints import extract_relative_pose_constraint
from src.slam.pose_graph.graph_builder import build_pose_graph
from src.slam.pose_graph.optimizer import optimize_pose_graph


def solve_bundle_windows_and_extract_constraints(
    slam_db: SlamDatabase,
    calibration,
    verbose: bool = True,
    bundle_solutions=None,
    keyframes=None,
):
    """Solve local BA windows and convert each optimized window into
    one relative keyframe constraint for the pose graph.

    If bundle_solutions and keyframes are provided (from a prior cache),
    they are used directly and solve_all_bundle_windows is skipped.
    """
    if bundle_solutions is None or keyframes is None:
        poses = [
            slam_db.manager_poses.get_pose(i)
            for i in range(len(slam_db.manager_poses.get_all_poses()))
        ]
        keyframes = choose_keyframes_by_motion(poses=np.array(poses))
        if verbose:
            print(f"Selected {len(keyframes)} motion-based keyframes.")
            print(f"First keyframes: {keyframes[:10]}")
            print(f"Last keyframes:  {keyframes[-10:]}")
        bundle_solutions = solve_all_bundle_windows(
            slam_db=slam_db,
            calibration=calibration,
            keyframes=keyframes,
            verbose=verbose,
        )
    else:
        if verbose:
            print(
                f"Reusing {len(bundle_solutions)} pre-computed BA window solutions ({len(keyframes)} keyframes)."
            )

    constraints = [extract_relative_pose_constraint(solution) for solution in bundle_solutions]

    return bundle_solutions, constraints


def _build_pg_no_lc(bundle_solutions=None, keyframes=None):
    """Section 7.1: detect loop-closure candidates using an empirically
    calibrated Mahalanobis threshold.

    The threshold was selected offline as the smallest value that retained
    at least 90% of temporally nonlocal pairs within 5 m and 20 degrees,
    while keeping the number of consensus-matching attempts small.

    Args:
        bundle_solutions: Pre-computed BA window solutions (from bundle_windows cache).
            If provided, skips re-running solve_all_bundle_windows.
        keyframes: Keyframe indices corresponding to bundle_solutions.
    """
    slam_db, calibration = build_database()
    _, constraints = solve_bundle_windows_and_extract_constraints(
        slam_db=slam_db,
        calibration=calibration,
        verbose=True,
        bundle_solutions=bundle_solutions,
        keyframes=keyframes,
    )
    if not constraints:
        raise RuntimeError("No relative-pose constraints were extracted.")
    first_frame = constraints[0].start_frame
    first_pose = pose3_from_world_to_camera_extrinsic(slam_db.manager_poses.get_pose(first_frame))
    print("Building graph...")
    build_result = build_pose_graph(constraints=constraints, first_pose=first_pose)
    print("Optimizing...")
    pose_graph_result = optimize_pose_graph(
        graph=build_result.graph,
        initial_estimates=build_result.initial_estimates,
        keyframe_ids=build_result.keyframe_ids,
    )
    keyframe_ids = build_result.keyframe_ids
    print("\nScoring temporally nonlocal candidate pairs...")
    all_candidates: list[LoopClosureCandidate] = []
    for target_frame in keyframe_ids:
        all_candidates.extend(
            score_candidates_for_keyframe(
                optimized_values=pose_graph_result.optimized_estimates,
                covariance_graph=build_result.covariance_graph,
                keyframe_ids=keyframe_ids,
                target_frame=target_frame,
            )
        )
    if not all_candidates:
        raise RuntimeError("No temporally nonlocal candidate pairs were scored.")

    candidates_by_target: dict[int, list[LoopClosureCandidate]] = {}
    accepted_candidates: list[LoopClosureCandidate] = []
    for target_frame in keyframe_ids:
        candidates = detect_candidates_for_keyframe(
            optimized_values=pose_graph_result.optimized_estimates,
            covariance_graph=build_result.covariance_graph,
            keyframe_ids=keyframe_ids,
            target_frame=target_frame,
            mahalanobis_threshold=config.MAHALANOBIS_THRESHOLD,
        )
        if candidates:
            candidates_by_target[target_frame] = candidates
            accepted_candidates.extend(candidates)

    print(f"Accepted candidate pairs: {len(accepted_candidates)}")
    return (candidates_by_target, build_result, pose_graph_result, slam_db, keyframe_ids)


def _build_pg_with_lc(
    graph: gtsam.NonlinearFactorGraph,
    initial_estimates: gtsam.Values,
    keyframe_ids: list[int],
    estimates: list[RelativePoseEstimate],
):
    """Section 7.4: Add the resulting measurement to the pose graph and optimize it to update the trajectory estimate.
    We incrementally add loop closures and optimize to collect intermediate versions for q_5.
    """
    print("Updating the Pose Graph with Loop Closures...")
    updated_graph = graph.clone()
    current_estimates = initial_estimates
    versions = []
    marginals_no_lc = gtsam.Marginals(graph, current_estimates)
    versions.append(("Optimized Without LC", current_estimates, marginals_no_lc))
    for idx, estimate in enumerate(estimates):
        source_key = C(estimate.consensus_result.candidate.source_frame)
        target_key = C(estimate.consensus_result.candidate.target_frame)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(estimate.covariance)
        factor = gtsam.BetweenFactorPose3(
            source_key, target_key, estimate.relative_pose, noise_model
        )
        updated_graph.add(factor)
        if idx == 0 and len(estimates) > 1:
            temp_result = optimize_pose_graph(updated_graph, current_estimates, keyframe_ids)
            current_estimates = temp_result.optimized_estimates
            temp_marginals = gtsam.Marginals(updated_graph.clone(), current_estimates)
            versions.append((f"Optimized with {idx + 1} LC", current_estimates, temp_marginals))
        elif idx == len(estimates) // 2 and len(versions) < 3 and (len(estimates) > 2):
            temp_result = optimize_pose_graph(updated_graph, current_estimates, keyframe_ids)
            current_estimates = temp_result.optimized_estimates
            temp_marginals = gtsam.Marginals(updated_graph.clone(), current_estimates)
            versions.append((f"Optimized with {idx + 1} LCs", current_estimates, temp_marginals))
    final_result = optimize_pose_graph(
        graph=updated_graph, initial_estimates=current_estimates, keyframe_ids=keyframe_ids
    )
    final_marginals = gtsam.Marginals(updated_graph.clone(), final_result.optimized_estimates)
    versions.append(("Optimized with All LCs", final_result.optimized_estimates, final_marginals))
    while len(versions) < 4:
        versions.append(versions[-1])
    return (final_result, versions[:4])
