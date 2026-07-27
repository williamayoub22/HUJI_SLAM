from src.slam.loop_closure.candidates import detect_candidates_for_keyframe, score_candidates_for_keyframe, LoopClosureCandidate, select_spread_loop_closures
from src.slam.loop_closure.consensus import ConsensusMatchResult, ConsensusMatcher
from src.slam.loop_closure.refinement import refine_relative_pose_with_bundle_adjustment, RelativePoseEstimate
from src.slam.pose_graph.graph_builder import PoseGraphBuildResult
import numpy as np
import gtsam
from gtsam.symbol_shorthand import C
from src.slam import config


def _build_lc_candidates(candidates_by_target: dict[int, list[LoopClosureCandidate]]) -> list[ConsensusMatchResult]:
    """Section 7.2: verify Mahalanobis-selected candidates using four-view
    PnP/RANSAC consensus matching.
    """
    min_loop_inliers = 20
    matcher = ConsensusMatcher()
    candidates = sorted((candidate for target_candidates in candidates_by_target.values() for candidate in target_candidates), key=lambda candidate: candidate.mahalanobis_squared)
    print('\n' + '=' * 60)
    print('[7.2] Consensus Matching')
    print('=' * 60)
    print(f'Mahalanobis-selected candidates: {len(candidates)}')
    print(f'Minimum accepted RANSAC inliers: {min_loop_inliers}')
    print('Reprojection threshold:          2 px (Exercise 3 setting)')
    results: list[ConsensusMatchResult] = []
    for index, candidate in enumerate(candidates, start=1):
        print(f'\nCandidate {index}/{len(candidates)}: c_{candidate.source_frame} -> c_{candidate.target_frame} (d^2={candidate.mahalanobis_squared:.1f})')
        result = matcher.verify(candidate)
        results.append(result)
        print(f'  Ratio-test temporal matches: {result.num_temporal_matches}')
        print(f'  Four-view correspondences:   {result.num_four_view_matches}')
        print(f'  RANSAC inliers:              {result.num_inliers}')
        print(f'  Inlier ratio:                {100.0 * result.inlier_ratio:.1f}%')
        if result.success:
            print('  Result: ACCEPTED loop closure')
        else:
            print(f'  Result: rejected — {result.failure_reason}')
    verified_results = [result for result in results if result.success]
    robust_results = [result for result in verified_results if result.num_inliers >= config.MIN_REPRESENTATIVE_LOOP_INLIERS]
    selected_results = select_spread_loop_closures(robust_results)
    print('\n' + '=' * 60)
    print('Consensus-matching summary')
    print('=' * 60)
    print(f'Candidates tested:              {len(results)}')
    print(f'Verified loop closures:         {len(verified_results)}')
    print(f'Robust loop closures:           {len(robust_results)} (inliers >= {config.MIN_REPRESENTATIVE_LOOP_INLIERS})')
    print(f'Selected for BA refinement:     {len(selected_results)}')
    if selected_results:
        print('\nSelected loop closures for refinement:')
        for result in selected_results:
            candidate = result.candidate
            print(f'  c_{candidate.source_frame} -> c_{candidate.target_frame}: {result.num_inliers} inliers ({100.0 * result.inlier_ratio:.1f}%)')
    return selected_results

def _refine_lc(selected_results: list[ConsensusMatchResult]) -> list[RelativePoseEstimate]:
    """Section 7.3: refine the selected consensus matches using two-frame stereo BA
    and extract a relative-pose covariance for each loop constraint.
    """
    print('\n' + '=' * 60)
    print('[7.3] Relative Pose Estimation')
    print('=' * 60)
    print(f'Selected loop closures refined: {len(selected_results)}')
    print('Initialization:                source pose fixed at identity; target initialized from PnP/RANSAC')
    print('Covariance extraction:         marginal covariance of target pose after two-frame stereo BA')
    estimates: list[RelativePoseEstimate] = []
    for index, result in enumerate(selected_results, start=1):
        candidate = result.candidate
        print(f'\nLoop {index}/{len(selected_results)}: c_{candidate.source_frame} -> c_{candidate.target_frame}')
        estimate = refine_relative_pose_with_bundle_adjustment(result)
        covariance_std = np.sqrt(np.diag(estimate.covariance))
        covariance_condition = np.linalg.cond(estimate.covariance)
        print(f'  Inlier landmarks:       {estimate.num_landmarks}')
        print(f'  BA error:               {estimate.initial_error:.2f} -> {estimate.final_error:.2f}')
        print(f'  Relative-pose std:       {np.array2string(covariance_std, precision=5)}')
        print(f'  Covariance condition:    {covariance_condition:.2e}')
        if covariance_condition > config.MAX_REFINED_COVARIANCE_CONDITION:
            print(f'  Result: rejected refined loop closure — covariance condition {covariance_condition:.2e} > {config.MAX_REFINED_COVARIANCE_CONDITION:.1e}')
            continue
        estimates.append(estimate)
        print('  Result: accepted refined loop closure')
    return estimates