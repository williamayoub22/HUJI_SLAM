import gtsam
from gtsam.symbol_shorthand import C
from src.slam.pipeline.database_pipeline import build_database
from src.slam import config
from src.slam.loop_closure.candidates import LoopClosureCandidate, score_candidates_for_keyframe, detect_candidates_for_keyframe
from src.slam.loop_closure.refinement import RelativePoseEstimate
from src.slam.pose_graph.graph_builder import build_pose_graph
from src.slam.pose_graph.optimizer import optimize_pose_graph
from src.slam.geometry.stereo import pose3_from_world_to_camera_extrinsic
from src.slam.loop_closure.diagnostics import _print_mahalanobis_threshold_sweep, _show_candidate_pair
# solve BA windows and extract all constraints

import numpy as np

from src.slam.geometry.stereo import make_gtsam_stereo_calibration
from src.slam.bundle_adjustment.window_selection import choose_keyframes_by_motion
from src.slam.bundle_adjustment.window_solver import solve_all_bundle_windows
from src.slam.data.db_facade import SlamDatabase
from src.slam.pose_graph.constraints import extract_relative_pose_constraint


def solve_bundle_windows_and_extract_constraints(
    slam_db: SlamDatabase,
    calibration,
    verbose: bool = True,
):
    """Solve local BA windows and convert each optimized window into
    one relative keyframe constraint for the pose graph.
    """
    poses = [slam_db.manager_poses.get_pose(i) for i in range(len(slam_db.manager_poses.get_all_poses()))]
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

    constraints = [extract_relative_pose_constraint(solution) for solution in bundle_solutions]

    return bundle_solutions, constraints



def _build_pg_no_lc():
    """Section 7.1: detect loop-closure candidates using an empirically
    calibrated Mahalanobis threshold.

    The threshold was selected offline as the smallest value that retained
    at least 90% of temporally nonlocal pairs within 5 m and 20 degrees,
    while keeping the number of consensus-matching attempts small.
    """
    keyframe_step = 10
    slam_db, calibration = build_database()
    _, constraints = solve_bundle_windows_and_extract_constraints(slam_db=slam_db, calibration=calibration, verbose=True)
    if not constraints:
        raise RuntimeError('No relative-pose constraints were extracted.')
    first_frame = constraints[0].start_frame
    first_pose = pose3_from_world_to_camera_extrinsic(slam_db.manager_poses.get_pose(first_frame))
    print('Building graph...')
    build_result = build_pose_graph(constraints=constraints, first_pose=first_pose)
    print('Optimizing...')
    pose_graph_result = optimize_pose_graph(graph=build_result.graph, initial_estimates=build_result.initial_estimates, keyframe_ids=build_result.keyframe_ids)
    keyframe_ids = build_result.keyframe_ids
    print('\nScoring all temporally nonlocal candidate pairs...')
    all_candidates: list[LoopClosureCandidate] = []
    for target_frame in keyframe_ids:
        all_candidates.extend(score_candidates_for_keyframe(optimized_values=pose_graph_result.optimized_estimates, covariance_graph=build_result.covariance_graph, keyframe_ids=keyframe_ids, target_frame=target_frame))
    if not all_candidates:
        raise RuntimeError('No temporally nonlocal candidate pairs were scored.')
    _print_mahalanobis_threshold_sweep(all_candidates, max_translation_m=5.0, max_rotation_deg=20.0)
    print('\n' + '=' * 60)
    print('[7.1] Mahalanobis Loop-Closure Candidate Detection')
    print('=' * 60)
    print('Keyframe selection:           motion-based')
    print(f'Keyframes tested:             {len(keyframe_ids)}')
    print(f'Minimum temporal separation:  {config.MIN_KEYFRAME_SEPARATION}')
    print(f'Mahalanobis threshold:        {config.MAHALANOBIS_THRESHOLD:.1f}')
    print('Threshold rationale:          smallest empirical threshold with at least 90% proxy-close-pair recall')
    candidates_by_target: dict[int, list[LoopClosureCandidate]] = {}
    accepted_candidates: list[LoopClosureCandidate] = []
    for target_frame in keyframe_ids:
        candidates = detect_candidates_for_keyframe(optimized_values=pose_graph_result.optimized_estimates, covariance_graph=build_result.covariance_graph, keyframe_ids=keyframe_ids, target_frame=target_frame, mahalanobis_threshold=config.MAHALANOBIS_THRESHOLD)
        if candidates:
            candidates_by_target[target_frame] = candidates
            accepted_candidates.extend(candidates)
    accepted_candidates.sort(key=lambda candidate: candidate.mahalanobis_squared)
    print(f'Accepted candidate pairs:     {len(accepted_candidates)}')
    print(f'Targets with candidates:      {len(candidates_by_target)}')
    if candidates_by_target:
        print(f'Mean candidates / target:     {len(accepted_candidates) / len(candidates_by_target):.2f}')
    print('\nBest selected candidates:')
    for candidate in accepted_candidates[:20]:
        delta = np.asarray(candidate.relative_delta, dtype=float)
        translation_m = np.linalg.norm(delta[3:])
        rotation_deg = np.degrees(np.linalg.norm(delta[:3]))
        print(f'  c_{candidate.source_frame} -> c_{candidate.target_frame}: d^2={candidate.mahalanobis_squared:.1f}, translation={translation_m:.2f} m, rotation={rotation_deg:.2f} deg')
    if accepted_candidates:
        best_candidate = accepted_candidates[0]
        _show_candidate_pair(candidate=best_candidate, optimized_values=pose_graph_result.optimized_estimates, keyframe_ids=keyframe_ids, title_prefix='Best Mahalanobis loop-closure candidate')
    return (candidates_by_target, build_result, pose_graph_result, slam_db, keyframe_ids)

def _build_pg_with_lc(graph: gtsam.NonlinearFactorGraph, initial_estimates: gtsam.Values, keyframe_ids: list[int], estimates: list[RelativePoseEstimate]):
    """Section 7.4: Add the resulting measurement to the pose graph and optimize it to update the trajectory estimate.
    We incrementally add loop closures and optimize to collect intermediate versions for q_5.
    """
    print('\n' + '=' * 60)
    print('[7.4] Update the Pose Graph')
    print('=' * 60)
    updated_graph = graph.clone()
    current_estimates = initial_estimates
    versions = []
    marginals_no_lc = gtsam.Marginals(graph, current_estimates)
    versions.append(('Optimized Without LC', current_estimates, marginals_no_lc))
    for idx, estimate in enumerate(estimates):
        source_key = C(estimate.consensus_result.candidate.source_frame)
        target_key = C(estimate.consensus_result.candidate.target_frame)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(estimate.covariance)
        factor = gtsam.BetweenFactorPose3(source_key, target_key, estimate.relative_pose, noise_model)
        updated_graph.add(factor)
        if idx == 0 and len(estimates) > 1:
            temp_result = optimize_pose_graph(updated_graph, current_estimates, keyframe_ids)
            current_estimates = temp_result.optimized_estimates
            temp_marginals = gtsam.Marginals(updated_graph.clone(), current_estimates)
            versions.append((f'Optimized with {idx + 1} LC', current_estimates, temp_marginals))
        elif idx == len(estimates) // 2 and len(versions) < 3 and (len(estimates) > 2):
            temp_result = optimize_pose_graph(updated_graph, current_estimates, keyframe_ids)
            current_estimates = temp_result.optimized_estimates
            temp_marginals = gtsam.Marginals(updated_graph.clone(), current_estimates)
            versions.append((f'Optimized with {idx + 1} LCs', current_estimates, temp_marginals))
    final_result = optimize_pose_graph(graph=updated_graph, initial_estimates=current_estimates, keyframe_ids=keyframe_ids)
    final_marginals = gtsam.Marginals(updated_graph.clone(), final_result.optimized_estimates)
    versions.append(('Optimized with All LCs', final_result.optimized_estimates, final_marginals))
    while len(versions) < 4:
        versions.append(versions[-1])
    return (final_result, versions[:4])