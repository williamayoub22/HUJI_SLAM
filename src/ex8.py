from src.slam.ba.window_solver import solve_all_bundle_windows
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import pickle
from pathlib import Path
from time import perf_counter
import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import C, Q
from tqdm import tqdm
from src.slam.ba.gtsam_utils import (
    make_stereo_camera,
    pose3_from_world_to_camera_extrinsic,
    stereo_point_from_triplet,
    make_gtsam_stereo_calibration,
    stereo_image_distances,
)
from src.slam.ba.optimization import _validate_graph_keys
from src.slam.ba.results import BundleAdjustmentResult, ProjectionFactorMetadata, BundleWindowSolution
from src.slam.ba.window_selection import collect_window_tracks, bundle_windows_from_keyframes, choose_keyframes_by_motion
from src.slam.config import (
    SEQUENCE_DIR,
    CACHE_DIR,
    GT_POSES_PATH,
    GLOBAL_CAMERA_MATRICES_PATH
)
from src.slam.database.facade import SlamDatabase
from src.slam.features.detectors import extract_features
from src.slam.features.matching import match_and_filter, get_matched_points
from src.slam.geometry.ransac import ransac_pnp
from src.slam.geometry.transforms import to_homogeneous_transform
from src.slam.io.calibration import read_stereo_calibration
from src.slam.io.image_loader import read_images
from src.slam.pipeline.database_pipeline import _create_frame_data, _match_temporal_features
from src.slam.geometry.triangulation import triangulate_opencv
from src.slam.loop_closure.candidate_detection import (
    ConsensusMatcher,
    ConsensusMatchResult,
    LoopClosureCandidate,
    RelativePoseEstimate,
    detect_candidates_for_keyframe,
    refine_relative_pose_with_bundle_adjustment,
    score_candidates_for_keyframe,
    select_spread_loop_closures,
)
from src.slam.pose_graph.constraints import extract_relative_pose_constraint
from src.slam.pose_graph.graph_builder import build_pose_graph
from src.slam.pose_graph.optimization import optimize_pose_graph
from src.slam.visualization.ex7_plots import (
    plot_absolute_location_error,
    plot_consensus_match,
    plot_location_uncertainty,
    plot_pose_graph_comparisons,
    plot_pose_graphs_versions,
    positions_from_values,
)
from src.slam.pipeline.pose_graph_pipeline import solve_bundle_windows_and_extract_constraints

def build_database():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    P1, P2 = read_stereo_calibration()
    K = P1[:, :3]
    calibration = gtsam.Cal3_S2Stereo(P1[0,0], P1[1,1], 0.0, P1[0,2], P1[1,2], -P2[0,3] / P1[0,0])

    db_path = CACHE_DIR / "slam_db"
    full_db_path = CACHE_DIR / "slam_db.pkl"

    if full_db_path.exists():
        print(f"Loading database from {full_db_path}...")
        slam_db = SlamDatabase.load(str(db_path))
        return slam_db, calibration

    print("Building database from scratch...")
    
    slam_db = SlamDatabase()
    
    total_frames = len(list((SEQUENCE_DIR / "image_0").glob("*.png")))
    
    global_T = np.eye(4)
    slam_db.manager_poses.add_pose(0, global_T)
    global_camera_matrices = [global_T[:3, :].copy()]
    inlier_percentages = []
    
    prev_frame_data = _create_frame_data(
        frame_idx=0,
        feature_type="akaze",
        num_features=3000,
        ratio_threshold=0.85,
        deviation_threshold=2.0,
        P1=P1,
        P2=P2,
    )
    
    slam_db.manager_2d.add_frame(
        links=prev_frame_data.links,
        left_features=prev_frame_data.left_features,
    )
    
    for frame_idx in tqdm(range(1, total_frames), desc="Building SLAM Database"):
        cur_frame_data = _create_frame_data(
            frame_idx=frame_idx,
            feature_type="akaze",
            num_features=3000,
            ratio_threshold=0.85,
            deviation_threshold=2.0,
            P1=P1,
            P2=P2,
        )
        
        temporal_matches, temporal_inliers = _match_temporal_features(
            prev_frame=prev_frame_data,
            cur_frame=cur_frame_data,
            feature_type="akaze",
            K=K,
            P1=P1,
            P2=P2,
        )
        
        inlier_ratio = sum(temporal_inliers) / len(temporal_matches) if len(temporal_matches) > 0 else 0.0
        inlier_percentages.append(inlier_ratio)
        
        candidate_prev_3d = []
        candidate_prev_left = []
        candidate_prev_right = []
        candidate_cur_left = []
        candidate_cur_right = []
        
        for match in temporal_matches:
            if not np.isfinite(match.distance): continue
            prev_idx = match.queryIdx
            cur_idx = match.trainIdx
            
            if prev_idx < len(prev_frame_data.points_3d) and cur_idx < len(cur_frame_data.links):
                p3d = prev_frame_data.points_3d[prev_idx]
                if np.all(np.isfinite(p3d)) and 0 < p3d[2] < 300.0:
                    candidate_prev_3d.append(p3d)
                    candidate_prev_left.append(prev_frame_data.links[prev_idx].left_keypoint())
                    candidate_prev_right.append(prev_frame_data.links[prev_idx].right_keypoint())
                    candidate_cur_left.append(cur_frame_data.links[cur_idx].left_keypoint())
                    candidate_cur_right.append(cur_frame_data.links[cur_idx].right_keypoint())

        T_rel = np.eye(4)
        if len(candidate_prev_3d) >= 4:
            T_candidate, _ = ransac_pnp(
                np.asarray(candidate_prev_3d),
                np.asarray(candidate_cur_left),
                np.asarray(candidate_prev_left),
                np.asarray(candidate_prev_right),
                np.asarray(candidate_cur_right),
                K, P1, P2
            )
            if T_candidate is not None:
                T_rel = T_candidate
                
        step_T = to_homogeneous_transform(T_rel)
        global_T = step_T @ global_T
        slam_db.manager_poses.add_pose(frame_idx, global_T)
        global_camera_matrices.append(global_T[:3, :].copy())
        
        slam_db.manager_2d.add_frame(
            links=cur_frame_data.links,
            left_features=cur_frame_data.left_features,
            matches_to_previous_left=temporal_matches,
            inliers=temporal_inliers,
        )
        
        prev_frame_data = cur_frame_data
        
    print("Triangulating 3D points...")
    for track_id in tqdm(slam_db.manager_2d.all_tracks(), desc="Triangulating points"):
        track_frames = slam_db.manager_2d.frames(track_id)
        if len(track_frames) < 2:
            continue
            
        initialization_frame = track_frames[-1]
        T_world_to_camera = slam_db.manager_poses.get_pose(initialization_frame)
        
        xl, xr, y = slam_db.manager_2d.link_triplet(initialization_frame, track_id)
        left_pt = np.array([[xl, y]])
        right_pt = np.array([[xr, y]])
        
        point_3d_camera = triangulate_opencv(P1, P2, left_pt, right_pt)[0]
        
        T_camera_to_world = np.linalg.inv(T_world_to_camera)
        point_4d = np.append(point_3d_camera, 1.0)
        point_3d_world = (T_camera_to_world @ point_4d)[:3]
        
        slam_db.manager_3d.add_point(track_id, point_3d_world)

    db_path = CACHE_DIR / "slam_db"
    slam_db.serialize(str(db_path))
    
    np.save(CACHE_DIR / "global_camera_matrices.npy", np.array(global_camera_matrices, dtype=float))
    np.save(CACHE_DIR / "inlier_percentages.npy", np.array(inlier_percentages, dtype=float))
    
    return slam_db, calibration

def load_or_build_db():
    full_db_path = CACHE_DIR / "slam_db.pkl"
    if full_db_path.exists():
        return SlamDatabase.load(str(CACHE_DIR / "slam_db"))
    return build_database()[0]

def load_or_build_pose_graph_no_lc():
    path = CACHE_DIR / "pose_graph_no_lc.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    # From q_1
    """Section 7.1: detect loop-closure candidates using an empirically
    calibrated Mahalanobis threshold.

    The threshold was selected offline as the smallest value that retained
    at least 90% of temporally nonlocal pairs within 5 m and 20 degrees,
    while keeping the number of consensus-matching attempts small.
    """
    min_keyframe_separation = 5
    keyframe_step = 10

    slam_db, calibration = build_database()

    _, constraints = solve_bundle_windows_and_extract_constraints(
        slam_db=slam_db,
        
        calibration=calibration,
        verbose=True,
    )

    if not constraints:
        raise RuntimeError("No relative-pose constraints were extracted.")

    first_frame = constraints[0].start_frame
    first_pose = pose3_from_world_to_camera_extrinsic(slam_db.manager_poses.get_pose(first_frame))

    print("Building graph...")
    build_result = build_pose_graph(
        constraints=constraints,
        first_pose=first_pose,
    )

    print("Optimizing...")
    pose_graph_result = optimize_pose_graph(
        graph=build_result.graph,
        initial_estimates=build_result.initial_estimates,
        keyframe_ids=build_result.keyframe_ids,
    )

    keyframe_ids = build_result.keyframe_ids

    # Offline study only: inspect every temporally nonlocal pair.
    print("\nScoring all temporally nonlocal candidate pairs...")
    all_candidates: list[LoopClosureCandidate] = []

    for target_frame in keyframe_ids:
        all_candidates.extend(
            score_candidates_for_keyframe(
                optimized_values=pose_graph_result.optimized_estimates,
                covariance_graph=build_result.covariance_graph,
                keyframe_ids=keyframe_ids,
                target_frame=target_frame,
                min_keyframe_separation=min_keyframe_separation,
            )
        )

    if not all_candidates:
        raise RuntimeError("No temporally nonlocal candidate pairs were scored.")

    _print_mahalanobis_threshold_sweep(
        all_candidates,
        max_translation_m=5.0,
        max_rotation_deg=20.0,
    )

    print("\n" + "=" * 60)
    print("[7.1] Mahalanobis Loop-Closure Candidate Detection")
    print("=" * 60)
    print("Keyframe selection:           motion-based")
    print(f"Keyframes tested:             {len(keyframe_ids)}")
    print(f"Minimum temporal separation:  {min_keyframe_separation}")
    print(f"Mahalanobis threshold:        {MAHALANOBIS_THRESHOLD:.1f}")
    print(
        "Threshold rationale:          smallest empirical threshold "
        "with at least 90% proxy-close-pair recall"
    )

    # Actual candidate-generation stage.
    candidates_by_target: dict[int, list[LoopClosureCandidate]] = {}
    accepted_candidates: list[LoopClosureCandidate] = []

    for target_frame in keyframe_ids:
        candidates = detect_candidates_for_keyframe(
            optimized_values=pose_graph_result.optimized_estimates,
            covariance_graph=build_result.covariance_graph,
            keyframe_ids=keyframe_ids,
            target_frame=target_frame,
            min_keyframe_separation=min_keyframe_separation,
            mahalanobis_threshold=MAHALANOBIS_THRESHOLD,
        )

        if candidates:
            candidates_by_target[target_frame] = candidates
            accepted_candidates.extend(candidates)

    accepted_candidates.sort(key=lambda candidate: candidate.mahalanobis_squared)

    print(f"Accepted candidate pairs:     {len(accepted_candidates)}")
    print(f"Targets with candidates:      {len(candidates_by_target)}")

    if candidates_by_target:
        print(
            "Mean candidates / target:     "
            f"{len(accepted_candidates) / len(candidates_by_target):.2f}"
        )

    print("\nBest selected candidates:")
    for candidate in accepted_candidates[:20]:
        delta = np.asarray(candidate.relative_delta, dtype=float)

        translation_m = np.linalg.norm(delta[3:])
        rotation_deg = np.degrees(np.linalg.norm(delta[:3]))

        print(
            f"  c_{candidate.source_frame} -> "
            f"c_{candidate.target_frame}: "
            f"d^2={candidate.mahalanobis_squared:.1f}, "
            f"translation={translation_m:.2f} m, "
            f"rotation={rotation_deg:.2f} deg"
        )

    if accepted_candidates:
        best_candidate = accepted_candidates[0]

        _show_candidate_pair(
            candidate=best_candidate,
            optimized_values=pose_graph_result.optimized_estimates,
            keyframe_ids=keyframe_ids,
            title_prefix="Best Mahalanobis loop-closure candidate",
        )

    data = {'candidates_by_target': candidates_by_target, 'build_result': build_result, 'pose_graph_result': pose_graph_result, 'keyframe_ids': keyframe_ids}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data

def load_or_build_loop_closures():
    path = CACHE_DIR / "loop_closures.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    pg_no_lc = load_or_build_pose_graph_no_lc()
    candidates_by_target = pg_no_lc["candidates_by_target"]
    
    # From q_2
    """Section 7.2: verify Mahalanobis-selected candidates using four-view
    PnP/RANSAC consensus matching.
    """
    min_loop_inliers = 20

    matcher = ConsensusMatcher(
        feature_type="akaze",
        min_loop_inliers=min_loop_inliers,
    )

    candidates = sorted(
        (
            candidate
            for target_candidates in candidates_by_target.values()
            for candidate in target_candidates
        ),
        key=lambda candidate: candidate.mahalanobis_squared,
    )

    print("\n" + "=" * 60)
    print("[7.2] Consensus Matching")
    print("=" * 60)
    print(f"Mahalanobis-selected candidates: {len(candidates)}")
    print(f"Minimum accepted RANSAC inliers: {min_loop_inliers}")
    print("Reprojection threshold:          2 px (Exercise 3 setting)")

    results: list[ConsensusMatchResult] = []

    for index, candidate in enumerate(candidates, start=1):
        print(
            f"\nCandidate {index}/{len(candidates)}: "
            f"c_{candidate.source_frame} -> c_{candidate.target_frame} "
            f"(d^2={candidate.mahalanobis_squared:.1f})"
        )

        result = matcher.verify(candidate)
        results.append(result)

        print(f"  Ratio-test temporal matches: {result.num_temporal_matches}")
        print(f"  Four-view correspondences:   {result.num_four_view_matches}")
        print(f"  RANSAC inliers:              {result.num_inliers}")
        print(f"  Inlier ratio:                {100.0 * result.inlier_ratio:.1f}%")

        if result.success:
            print("  Result: ACCEPTED loop closure")
        else:
            print(f"  Result: rejected — {result.failure_reason}")

    verified_results = [result for result in results if result.success]

    robust_results = [
        result
        for result in verified_results
        if result.num_inliers >= MIN_REPRESENTATIVE_LOOP_INLIERS
    ]

    selected_results = select_spread_loop_closures(
        robust_results,
        num_representatives=NUM_LOOP_CLOSURE_REPRESENTATIVES,
    )

    print("\n" + "=" * 60)
    print("Consensus-matching summary")
    print("=" * 60)
    print(f"Candidates tested:              {len(results)}")
    print(f"Verified loop closures:         {len(verified_results)}")
    print(
        "Robust loop closures:           "
        f"{len(robust_results)} "
        f"(inliers >= {MIN_REPRESENTATIVE_LOOP_INLIERS})"
    )
    print(f"Selected for BA refinement:     {len(selected_results)}")

    if selected_results:
        print("\nSelected loop closures for refinement:")
        for result in selected_results:
            candidate = result.candidate
            print(
                f"  c_{candidate.source_frame} -> c_{candidate.target_frame}: "
                f"{result.num_inliers} inliers "
                f"({100.0 * result.inlier_ratio:.1f}%)"
            )

    return selected_results

    # From q_3
    """Section 7.3: refine the selected consensus matches using two-frame stereo BA
    and extract a relative-pose covariance for each loop constraint.
    """
    print("\n" + "=" * 60)
    print("[7.3] Relative Pose Estimation")
    print("=" * 60)
    print(f"Selected loop closures refined: {len(selected_results)}")
    print(
        "Initialization:                source pose fixed at identity; "
        "target initialized from PnP/RANSAC"
    )
    print(
        "Covariance extraction:         "
        "marginal covariance of target pose after two-frame stereo BA"
    )

    estimates: list[RelativePoseEstimate] = []

    for index, result in enumerate(selected_results, start=1):
        candidate = result.candidate

        print(
            f"\nLoop {index}/{len(selected_results)}: "
            f"c_{candidate.source_frame} -> c_{candidate.target_frame}"
        )

        estimate = refine_relative_pose_with_bundle_adjustment(result)
        covariance_std = np.sqrt(np.diag(estimate.covariance))
        covariance_condition = np.linalg.cond(estimate.covariance)

        print(f"  Inlier landmarks:       {estimate.num_landmarks}")
        print(
            f"  BA error:               {estimate.initial_error:.2f} -> {estimate.final_error:.2f}"
        )
        print(f"  Relative-pose std:       {np.array2string(covariance_std, precision=5)}")
        print(f"  Covariance condition:    {covariance_condition:.2e}")

        if covariance_condition > MAX_REFINED_COVARIANCE_CONDITION:
            print(
                "  Result: rejected refined loop closure — "
                f"covariance condition {covariance_condition:.2e} "
                f"> {MAX_REFINED_COVARIANCE_CONDITION:.1e}"
            )
            continue

        estimates.append(estimate)
        print("  Result: accepted refined loop closure")

    return estimates

    # Custom stats return
    lc_stats = []
    for r in selected_results:
        if r.success:
            lc_stats.append({
                "target_frame": r.candidate.target_frame,
                "source_frame": r.candidate.source_frame,
                "num_matches": r.num_four_view_matches,
                "inlier_percentage": r.inlier_ratio * 100.0,
            })
            
    match_data = None
    successful = [r for r in selected_results if r.success]
    if successful:
        first = successful[0]
        match_data = {
            "source_frame": first.candidate.source_frame,
            "target_frame": first.candidate.target_frame,
            "source_left": first.source_left,
            "target_left": first.target_left,
            "source_inliers": first.source_inliers,
            "target_inliers": first.target_inliers,
            "source_outliers": first.source_outliers,
            "target_outliers": first.target_outliers,
        }
        
    data = {
        "estimates": estimates,
        "lc_stats": lc_stats,
        "match_data": match_data,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data

def load_or_build_pose_graph_with_lc():
    path = CACHE_DIR / "pose_graph_with_lc.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    pg_no_lc = load_or_build_pose_graph_no_lc()
    lc = load_or_build_loop_closures()
    
    graph = pg_no_lc["build_result"].graph
    initial_estimates = pg_no_lc["pose_graph_result"].optimized_estimates
    keyframe_ids = pg_no_lc["keyframe_ids"]
    estimates = lc["estimates"]
    
    # From q_4
    """Section 7.4: Add the resulting measurement to the pose graph and optimize it to update the trajectory estimate.
    We incrementally add loop closures and optimize to collect intermediate versions for q_5.
    """
    print("\n" + "=" * 60)
    print("[7.4] Update the Pose Graph")
    print("=" * 60)

    updated_graph = graph.clone()
    current_estimates = initial_estimates
    versions = []

    # 1. Use the original, unmutated graph for the 'Without LC' marginals
    marginals_no_lc = gtsam.Marginals(graph, current_estimates)
    versions.append(("Optimized Without LC", current_estimates, marginals_no_lc))

    for idx, estimate in enumerate(estimates):
        source_key = C(estimate.consensus_result.candidate.source_frame)
        target_key = C(estimate.consensus_result.candidate.target_frame)

        # 1. Use the EXACT covariance from the refinement. Do NOT inflate it.
        # This gives the loop closure the mathematical weight needed to bend the graph.
        noise_model = gtsam.noiseModel.Gaussian.Covariance(estimate.covariance)

        factor = gtsam.BetweenFactorPose3(
            source_key, target_key, estimate.relative_pose, noise_model
        )
        updated_graph.add(factor)

        if idx == 0 and len(estimates) > 1:
            temp_result = optimize_pose_graph(updated_graph, current_estimates, keyframe_ids)
            current_estimates = temp_result.optimized_estimates
            # Freeze the graph state for this marginal by cloning it
            temp_marginals = gtsam.Marginals(updated_graph.clone(), current_estimates)
            versions.append((f"Optimized with {idx + 1} LC", current_estimates, temp_marginals))
        elif idx == len(estimates) // 2 and len(versions) < 3 and len(estimates) > 2:
            temp_result = optimize_pose_graph(updated_graph, current_estimates, keyframe_ids)
            current_estimates = temp_result.optimized_estimates
            # Freeze the graph state for this marginal by cloning it
            temp_marginals = gtsam.Marginals(updated_graph.clone(), current_estimates)
            versions.append((f"Optimized with {idx + 1} LCs", current_estimates, temp_marginals))

    final_result = optimize_pose_graph(
        graph=updated_graph,
        initial_estimates=current_estimates,
        keyframe_ids=keyframe_ids,
    )

    # Freeze the final graph state
    final_marginals = gtsam.Marginals(updated_graph.clone(), final_result.optimized_estimates)
    versions.append(("Optimized with All LCs", final_result.optimized_estimates, final_marginals))

    while len(versions) < 4:
        versions.append(versions[-1])

    return final_result, versions[:4]

    versions_data = []
    for title, values, marginals in versions:
        positions = positions_from_values(values, keyframe_ids)
        covs = {}
        if marginals is not None:
            for fid in keyframe_ids:
                covs[fid] = np.asarray(marginals.marginalCovariance(gtsam.symbol("c", fid)))
        versions_data.append({
            "title": title,
            "positions": positions,
            "covs": covs,
        })
        
    pose_graph_matrices = [
        versions[0][1].atPose3(gtsam.symbol("c", fid)).matrix()
        for fid in keyframe_ids
    ]
    pose_graph_lc_matrices = [
        versions[-1][1].atPose3(gtsam.symbol("c", fid)).matrix()
        for fid in keyframe_ids
    ]
    
    data = {
        "updated_pose_graph_result": pose_graph_result,
        "pose_graph_matrices": pose_graph_matrices,
        "pose_graph_lc_matrices": pose_graph_lc_matrices,
        "versions_data": versions_data,
    }
    
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data
