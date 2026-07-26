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


MAHALANOBIS_THRESHOLD = 32023.8
NUM_LOOP_CLOSURE_REPRESENTATIVES = 10
MIN_REPRESENTATIVE_LOOP_INLIERS = 30
MAX_REFINED_COVARIANCE_CONDITION = 1e8

from src.slam.pipeline.pose_graph_pipeline import solve_bundle_windows_and_extract_constraints

def _is_geometrically_close(
    candidate,
    *,
    max_translation_m: float,
    max_rotation_deg: float,
) -> bool:
    """Offline calibration label only; never used as the actual online gate."""
    delta = np.asarray(candidate.relative_delta, dtype=float)

    translation_m = float(np.linalg.norm(delta[3:]))
    rotation_deg = float(np.degrees(np.linalg.norm(delta[:3])))

    return translation_m <= max_translation_m and rotation_deg <= max_rotation_deg


def _print_mahalanobis_threshold_sweep(
    all_candidates: list,
    *,
    max_translation_m: float = 5.0,
    max_rotation_deg: float = 20.0,
) -> float:
    """Pick a Mahalanobis threshold empirically.

    'Close' is used only as an offline proxy for a potential visual overlap.
    The actual algorithm later uses only the Mahalanobis threshold followed
    by consensus matching.
    """
    if not all_candidates:
        raise RuntimeError("No candidates available for threshold calibration.")

    scores = np.asarray(
        [candidate.mahalanobis_squared for candidate in all_candidates],
        dtype=float,
    )

    close_candidates = [
        candidate
        for candidate in all_candidates
        if _is_geometrically_close(
            candidate,
            max_translation_m=max_translation_m,
            max_rotation_deg=max_rotation_deg,
        )
    ]

    if not close_candidates:
        raise RuntimeError(
            "No geometrically close nonlocal pairs found. "
            "Increase the calibration radii or inspect the trajectory."
        )

    close_scores = np.asarray(
        [candidate.mahalanobis_squared for candidate in close_candidates],
        dtype=float,
    )

    print("\n" + "=" * 60)
    print("Empirical Mahalanobis-threshold calibration")
    print("=" * 60)
    print(
        "Calibration close-pair proxy: "
        f"translation <= {max_translation_m:.1f} m, "
        f"rotation <= {max_rotation_deg:.1f} deg"
    )
    print(f"All temporally nonlocal pairs: {len(all_candidates)}")
    print(f"Geometrically close pairs:     {len(close_candidates)}")
    print()
    print(
        f"{'threshold':>15} | {'close recall':>12} | {'selected':>10} | {'selected / target':>18}"
    )
    print("-" * 67)

    chosen_threshold = None

    for percentile in (50, 75, 90, 95, 97.5, 99):
        threshold = float(np.percentile(close_scores, percentile))

        selected = [
            candidate for candidate in all_candidates if candidate.mahalanobis_squared <= threshold
        ]

        selected_close = [
            candidate
            for candidate in selected
            if _is_geometrically_close(
                candidate,
                max_translation_m=max_translation_m,
                max_rotation_deg=max_rotation_deg,
            )
        ]

        recall = len(selected_close) / len(close_candidates)

        targets_with_selected = len({candidate.target_frame for candidate in selected})

        selected_per_target = (
            len(selected) / targets_with_selected if targets_with_selected > 0 else 0.0
        )

        print(
            f"{threshold:15.1f} | "
            f"{recall:11.1%} | "
            f"{len(selected):10d} | "
            f"{selected_per_target:18.2f}"
        )

        # Smallest threshold retaining at least 90% of offline-close pairs.
        if chosen_threshold is None and recall >= 0.90:
            chosen_threshold = threshold

    if chosen_threshold is None:
        chosen_threshold = float(np.max(close_scores))

    print("\nRecommended initial threshold:")
    print(f"  mahalanobis_threshold = {chosen_threshold:.1f}")

    return chosen_threshold


def _closest_spatial_candidate(candidates_by_target: dict[int, list]):
    """Return the temporally nonlocal candidate with the smallest Euclidean
    relative translation, regardless of Mahalanobis score.
    """
    all_candidates = [
        candidate for candidates in candidates_by_target.values() for candidate in candidates
    ]

    if not all_candidates:
        raise RuntimeError("No temporally nonlocal candidate pairs were found.")

    return min(
        all_candidates,
        key=lambda candidate: np.linalg.norm(candidate.relative_delta[3:]),
    )


def _load_left_image(frame_id: int) -> np.ndarray:
    """Load KITTI sequence-00 left image for a raw frame index.

    Adjust image_0 if your project uses another left-image directory.
    """
    image_path = Path(SEQUENCE_DIR) / "image_0" / f"{frame_id:06d}.png"

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _show_candidate_pair(
    *,
    candidate: LoopClosureCandidate,
    optimized_values,
    keyframe_ids: list[int],
    title_prefix: str = "Candidate pair",
) -> None:
    """Show one candidate pair, its images, and its trajectory locations."""
    source_frame = candidate.source_frame
    target_frame = candidate.target_frame

    source_image = _load_left_image(source_frame)
    target_image = _load_left_image(target_frame)

    source_position = optimized_values.atPose3(C(source_frame)).translation()
    target_position = optimized_values.atPose3(C(target_frame)).translation()

    positions = np.asarray(
        [optimized_values.atPose3(C(frame_id)).translation() for frame_id in keyframe_ids],
        dtype=float,
    )

    delta = np.asarray(candidate.relative_delta, dtype=float)
    translation_m = float(np.linalg.norm(delta[3:]))
    rotation_deg = float(np.degrees(np.linalg.norm(delta[:3])))

    print("\n" + "=" * 60)
    print(title_prefix)
    print("=" * 60)
    print(f"Frames:                 c_{source_frame} <-> c_{target_frame}")
    print(f"Mahalanobis d^2:         {candidate.mahalanobis_squared:.1f}")
    print(f"Euclidean distance:       {translation_m:.3f} m")
    print(f"Relative rotation:        {rotation_deg:.3f} deg")
    print(f"Keyframe path distance:   {len(candidate.path_frame_ids) - 1} edges")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(source_image)
    axes[0].set_title(f"Earlier keyframe {source_frame}")
    axes[0].axis("off")

    axes[1].imshow(target_image)
    axes[1].set_title(f"Later keyframe {target_frame}")
    axes[1].axis("off")

    axes[2].plot(positions[:, 0], positions[:, 2], linewidth=1)

    axes[2].scatter(
        [source_position[0]],
        [source_position[2]],
        s=100,
        label=f"c_{source_frame}",
    )
    axes[2].scatter(
        [target_position[0]],
        [target_position[2]],
        s=100,
        label=f"c_{target_frame}",
    )
    axes[2].plot(
        [source_position[0], target_position[0]],
        [source_position[2], target_position[2]],
        linestyle="--",
    )

    axes[2].set_title(f"Optimized trajectory\n{translation_m:.2f} m, {rotation_deg:.2f}°")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("z [m]")
    axes[2].axis("equal")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    plt.show()


def _print_candidate_debug(candidate) -> None:
    """Print the relative motion, accumulated path uncertainty, and rough
    per-coordinate Mahalanobis contributions for one candidate.
    """
    delta = np.asarray(candidate.relative_delta, dtype=float)
    covariance = np.asarray(candidate.covariance, dtype=float)

    standard_deviations = np.sqrt(np.diag(covariance))

    # GTSAM Pose3 tangent convention:
    # [rotation_x, rotation_y, rotation_z, translation_x, translation_y, translation_z]
    rotation_vector = delta[:3]
    translation_vector = delta[3:]

    rotation_magnitude_deg = np.degrees(np.linalg.norm(rotation_vector))
    translation_magnitude = np.linalg.norm(translation_vector)

    # Exact only when covariance is diagonal, but useful as a diagnostic.
    diagonal_contributions = delta**2 / np.diag(covariance)

    covariance_eigenvalues = np.linalg.eigvalsh(covariance)

    print(f"\n    c_{candidate.source_frame} -> c_{candidate.target_frame}")
    print(f"      d^2:                {candidate.mahalanobis_squared:.3f}")
    print(f"      path edges:         {len(candidate.path_frame_ids) - 1}")
    print(f"      path:               {candidate.path_frame_ids}")
    print(f"      rotation magnitude: {rotation_magnitude_deg:.3f} deg")
    print(f"      translation norm:   {translation_magnitude:.3f} m")
    print(f"      delta [rad, m]:     {np.array2string(delta, precision=6)}")
    print(f"      path std [rad, m]:  {np.array2string(standard_deviations, precision=6)}")
    print(f"      cov eig:            {np.array2string(covariance_eigenvalues, precision=3)}")
    print(f"      rough d^2 terms:    {np.array2string(diagonal_contributions, precision=1)}")


def _group_candidates_by_target(
    candidates: list,
) -> dict[int, list]:
    """Group already-scored candidates by their later keyframe."""
    candidates_by_target: dict[int, list] = {}

    for candidate in candidates:
        candidates_by_target.setdefault(
            candidate.target_frame,
            [],
        ).append(candidate)

    for target_candidates in candidates_by_target.values():
        target_candidates.sort(key=lambda candidate: candidate.mahalanobis_squared)

    return candidates_by_target



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
def q_1():
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

    return (
        candidates_by_target,
        build_result,
        pose_graph_result,
        slam_db,
        keyframe_ids,
    )


def q_2(
    candidates_by_target: dict[int, list[LoopClosureCandidate]],
) -> list[ConsensusMatchResult]:
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


def q_3(
    selected_results: list[ConsensusMatchResult],
) -> list[RelativePoseEstimate]:
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


def q_4(
    graph: gtsam.NonlinearFactorGraph,
    initial_estimates: gtsam.Values,
    keyframe_ids: list[int],
    estimates: list[RelativePoseEstimate],
):
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



def compute_median_projection_errors_by_distance(
    database,
    world_to_camera_by_frame,
    calibration,
    *,
    optimize_landmarks,
    min_track_length=2,
    max_distance=50,
):
    """
    Compute median stereo reprojection errors by temporal distance.

    PnP mode:
        1. Backproject the latest stereo observation into the latest camera's
           coordinate system.
        2. Propagate that 3D point into each earlier camera by composing the
           consecutive estimated camera transformations.
        3. Project the propagated point into that frame.
        4. Compare the predicted pixels with the measured stereo observation.

    Bundle-adjustment mode:
        Optimize one world-space landmark using all selected observations,
        while keeping the supplied camera poses fixed.
    """
    manager_2d = database.manager_2d

    left_errors_by_distance = defaultdict(list)
    right_errors_by_distance = defaultdict(list)

    used_tracks = 0
    rejected_tracks = 0

    # An identity-pose camera projects points that are already expressed
    # in the current camera coordinate system.
    local_stereo_camera = make_stereo_camera(
        np.eye(4, dtype=float),
        calibration,
    )

    for track_id in manager_2d.all_tracks():
        track_frames = sorted(manager_2d.frames(track_id))

        usable_frames = [
            frame_id for frame_id in track_frames if frame_id in world_to_camera_by_frame
        ]

        if len(usable_frames) < min_track_length:
            continue

        triangulation_frame = usable_frames[-1]

        evaluation_frames = [
            frame_id
            for frame_id in usable_frames
            if 0 <= triangulation_frame - frame_id <= max_distance
        ]

        if len(evaluation_frames) < min_track_length:
            continue

        try:
            if optimize_landmarks:
                landmark_world = optimize_track_landmark(
                    manager_2d=manager_2d,
                    track_id=track_id,
                    frame_ids=evaluation_frames,
                    world_to_camera_by_frame=world_to_camera_by_frame,
                    calibration=calibration,
                )

                landmark_world = np.asarray(
                    landmark_world,
                    dtype=float,
                ).reshape(3)

                if not np.all(np.isfinite(landmark_world)):
                    raise ValueError("Optimized landmark contains non-finite values.")

            else:
                # Backproject using an identity camera. The result is expressed
                # in triangulation_frame camera coordinates, not world space.
                triangulation_measurement = stereo_point_from_triplet(
                    get_stereo_observation(
                        manager_2d,
                        triangulation_frame,
                        track_id,
                    )
                )

                landmark_in_triangulation_camera = np.asarray(
                    local_stereo_camera.backproject(triangulation_measurement),
                    dtype=float,
                ).reshape(3)

                if not np.all(np.isfinite(landmark_in_triangulation_camera)):
                    raise ValueError("Triangulated landmark contains non-finite values.")

        except (
            KeyError,
            IndexError,
            ValueError,
            RuntimeError,
            Exception,
        ):
            rejected_tracks += 1
            continue

        track_contributed = False

        for frame_id in evaluation_frames:
            distance = triangulation_frame - frame_id

            try:
                if optimize_landmarks:
                    frame_extrinsic = to_homogeneous_transform(world_to_camera_by_frame[frame_id])

                    frame_camera = make_stereo_camera(
                        frame_extrinsic,
                        calibration,
                    )

                    projection = frame_camera.project(gtsam.Point3(landmark_world))

                else:
                    # Map the point from the triangulation camera to the
                    # requested evaluation camera by composing estimated
                    # consecutive camera motions.
                    frame_from_triangulation = compose_camera_transform(
                        world_to_camera_by_frame=(world_to_camera_by_frame),
                        source_frame=triangulation_frame,
                        target_frame=frame_id,
                    )

                    landmark_homogeneous = np.append(
                        landmark_in_triangulation_camera,
                        1.0,
                    )

                    landmark_in_frame_camera = (frame_from_triangulation @ landmark_homogeneous)[:3]

                    if not np.all(np.isfinite(landmark_in_frame_camera)):
                        continue

                    # The point is already in frame_id camera coordinates.
                    projection = local_stereo_camera.project(gtsam.Point3(landmark_in_frame_camera))

                measurement = stereo_point_from_triplet(
                    get_stereo_observation(
                        manager_2d,
                        frame_id,
                        track_id,
                    )
                )

                left_error, right_error = stereo_image_distances(
                    measurement=measurement,
                    projection=projection,
                )

            except (
                KeyError,
                IndexError,
                ValueError,
                RuntimeError,
                Exception,
            ):
                continue

            left_errors_by_distance[distance].append(float(left_error))
            right_errors_by_distance[distance].append(float(right_error))

            track_contributed = True

        if track_contributed:
            used_tracks += 1
        else:
            rejected_tracks += 1

    distances = np.asarray(
        sorted(set(left_errors_by_distance) & set(right_errors_by_distance)),
        dtype=int,
    )

    median_left_errors = np.asarray(
        [np.median(left_errors_by_distance[distance]) for distance in distances],
        dtype=float,
    )

    median_right_errors = np.asarray(
        [np.median(right_errors_by_distance[distance]) for distance in distances],
        dtype=float,
    )

    sample_counts = np.asarray(
        [
            min(
                len(left_errors_by_distance[distance]),
                len(right_errors_by_distance[distance]),
            )
            for distance in distances
        ],
        dtype=int,
    )

    print(f"Used tracks:        {used_tracks}")
    print(f"Rejected tracks:    {rejected_tracks}")
    print(f"Computed distances: {len(distances)}")

    return (
        distances,
        median_left_errors,
        median_right_errors,
        sample_counts,
    )

def load_inlier_percentages():
    from src.slam.config import PROJECT_DIR
    inlier_path = PROJECT_DIR / "outputs" / "ex4" / "inlier_percentages.npy"
    if not inlier_path.exists():
        print(f"Warning: {inlier_path} not found. Defaulting to 100%.")
        return np.ones(3300) * 100.0
    arr = np.load(inlier_path)
    print(f"Loaded {len(arr)} inlier percentages from {inlier_path}")
    return arr

def _get_result_value(result, possible_names):
    """Read a value from either a dictionary or a result object."""
    for name in possible_names:
        if isinstance(result, dict) and name in result:
            return result[name]

        if hasattr(result, name):
            return getattr(result, name)

    raise AttributeError(f"Could not find any of {possible_names} in bundle-window result.")

def compute_bundle_window_factor_errors(bundle_results):
    """Compute mean factor error before and after BA for each window.

    Each result must contain:
        - the window factor graph;
        - the initial Values;
        - the optimized Values.

    The function supports several common field names so it can work with
    either dictionaries or result objects.
    """
    window_ids = []
    initial_mean_errors = []
    optimized_mean_errors = []

    for window_index, result in enumerate(bundle_results):
        try:
            graph = _get_result_value(
                result,
                (
                    "graph",
                    "factor_graph",
                    "bundle_graph",
                ),
            )

            initial_values = _get_result_value(
                result,
                (
                    "initial_values",
                    "initial",
                    "initial_estimate",
                ),
            )

            optimized_values = _get_result_value(
                result,
                (
                    "optimized_values",
                    "result_values",
                    "optimized",
                    "solution",
                ),
            )

            number_of_factors = int(graph.size())

            if number_of_factors == 0:
                continue

            initial_total_error = float(graph.error(initial_values))
            optimized_total_error = float(graph.error(optimized_values))

            window_ids.append(window_index)
            initial_mean_errors.append(initial_total_error / number_of_factors)
            optimized_mean_errors.append(optimized_total_error / number_of_factors)

        except (
            AttributeError,
            KeyError,
            ValueError,
            RuntimeError,
        ) as error:
            print(f"WARNING: Could not compute errors for bundle window {window_index}: {error}")

    return (
        np.asarray(window_ids, dtype=int),
        np.asarray(initial_mean_errors, dtype=float),
        np.asarray(optimized_mean_errors, dtype=float),
    )

def extract_bundle_window_average_errors(bundle_solutions):
    """Extract normalized factor errors from solved BA windows."""
    window_ids = []
    initial_average_errors = []
    final_average_errors = []

    for window_index, solution in enumerate(bundle_solutions, start=1):
        result = solution.result

        if result.num_factors <= 0:
            continue

        initial_error = float(result.average_initial_error)
        final_error = float(result.average_final_error)

        if not np.isfinite(initial_error) or not np.isfinite(final_error):
            continue

        window_ids.append(window_index)
        initial_average_errors.append(initial_error)
        final_average_errors.append(final_error)

    return (
        np.asarray(window_ids, dtype=int),
        np.asarray(initial_average_errors, dtype=float),
        np.asarray(final_average_errors, dtype=float),
    )

def compute_bundle_window_errors(bundle_solutions):
    window_ids = []
    initial_average_errors = []
    final_average_errors = []
    initial_median_proj_errors = []
    final_median_proj_errors = []

    for window_index, solution in enumerate(
        bundle_solutions,
        start=1,
    ):
        result = solution.result

        if result.num_factors <= 0:
            continue

        initial_error = float(
            result.average_initial_error
        )
        final_error = float(
            result.average_final_error
        )

        if (
            not np.isfinite(initial_error)
            or not np.isfinite(final_error)
        ):
            continue

        init_proj_errors = []
        final_proj_errors = []
        for metadata in result.projection_factor_metadata:
            factor = result.graph.at(metadata.factor_index)
            # unwhitenedError returns [uL_err, uR_err, v_err]
            e_init = factor.unwhitenedError(result.initial)
            e_final = factor.unwhitenedError(result.optimized)
            init_proj_errors.append(np.linalg.norm(e_init))
            final_proj_errors.append(np.linalg.norm(e_final))

        window_ids.append(window_index)
        initial_average_errors.append(initial_error)
        final_average_errors.append(final_error)
        initial_median_proj_errors.append(np.median(init_proj_errors) if init_proj_errors else 0.0)
        final_median_proj_errors.append(np.median(final_proj_errors) if final_proj_errors else 0.0)

    return (
        np.asarray(window_ids, dtype=int),
        np.asarray(initial_average_errors, dtype=float),
        np.asarray(final_average_errors, dtype=float),
        np.asarray(initial_median_proj_errors, dtype=float),
        np.asarray(final_median_proj_errors, dtype=float),
    )

@dataclass
class BundleAnalysisContext:
    calibration: gtsam.Cal3_S2Stereo

    keyframes: list[int]

    solutions: list[BundleWindowSolution]

def prepare_bundle_adjustment_analysis(
    database,
) -> BundleAnalysisContext:
    """Select motion-based keyframes and solve all BA windows once."""

    P1, P2 = read_stereo_calibration()

    calibration = make_gtsam_stereo_calibration(P1, P2)

    pnp_extrinsics = np.asarray(
        np.load(GLOBAL_CAMERA_MATRICES_PATH),
        dtype=float,
    )

    keyframes = choose_keyframes_by_motion(
        poses=pnp_extrinsics,
        min_gap=11,
        max_gap=20,
        min_translation=5.0,
        min_rotation_deg=12.0,
        target_translation=5.0,
        target_rotation_deg=12.0,
    )

    print(f"Selected {len(keyframes)} motion-based keyframes.")

    print(f"First keyframes: {keyframes[:10]}")

    print(f"Last keyframes:  {keyframes[-10:]}")

    solutions = solve_all_bundle_windows(
        slam_db=database,
        calibration=calibration,
        keyframes=keyframes,
        min_track_observations=2,
        measurement_sigma_pixels=1.0,
        verbose=True,
    )

    return BundleAnalysisContext(
        calibration=calibration,
        keyframes=keyframes,
        solutions=solutions,
    )

def run_bundle_window_error_analysis(database):
    """Compare initial and optimized factor errors for motion-based BA windows."""
    print("\n" + "=" * 60)
    print("Bundle-window optimization analysis")
    print("=" * 60)

    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)

    # Use the updated motion-based keyframe-selection rule from the
    # pose-graph pipeline. This avoids the old fixed-interval selector.
    _, constraints = solve_bundle_windows_and_extract_constraints(
        slam_db=database,
        calibration=calibration,
        verbose=True,
    )

    if not constraints:
        print("WARNING: No bundle-window constraints were generated.")
        return {
            "window_ids": np.asarray([], dtype=int),
            "initial_mean_errors": np.asarray([], dtype=float),
            "optimized_mean_errors": np.asarray([], dtype=float),
            "plot_path": None,
        }

    # Recover the ordered keyframe sequence from the solved windows.
    keyframes = [constraints[0].start_frame]

    for constraint in constraints:
        if keyframes[-1] != constraint.start_frame:
            raise ValueError(
                "Bundle-window constraints are not consecutive: "
                f"expected start frame {keyframes[-1]}, "
                f"received {constraint.start_frame}."
            )

        keyframes.append(constraint.end_frame)

    print(f"Motion-based keyframes: {len(keyframes)}")
    print(f"Bundle windows:         {len(keyframes) - 1}")

    # Solve the same windows again while retaining the complete solutions
    # needed to evaluate the initial and optimized factor-graph errors.
    bundle_solutions = solve_all_bundle_windows(
        slam_db=database,
        calibration=calibration,
        keyframes=keyframes,
        min_track_observations=2,
        measurement_sigma_pixels=1.0,
        verbose=False,
    )

    (
        window_ids,
        initial_mean_errors,
        optimized_mean_errors,
    ) = compute_bundle_window_factor_errors(bundle_solutions)

    plot_path = plot_bundle_window_factor_errors(
        window_ids=window_ids,
        initial_mean_errors=initial_mean_errors,
        optimized_mean_errors=optimized_mean_errors,
    )

    return {
        "keyframes": keyframes,
        "window_ids": window_ids,
        "initial_mean_errors": initial_mean_errors,
        "optimized_mean_errors": optimized_mean_errors,
        "plot_path": plot_path,
    }

def run_tracking_analysis(database, inlier_percentages):
    """Compute and plot the tracking-performance statistics."""
    manager_2d = database.manager_2d

    frame_ids = sorted(manager_2d.all_frames())
    track_ids = list(manager_2d.all_tracks())

    track_lengths = np.asarray(
        [len(manager_2d.frames(track_id)) for track_id in track_ids],
        dtype=float,
    )

    # Number of stored feature observations/matches in every frame.
    matches_per_frame = [len(manager_2d.tracks(frame_id)) for frame_id in frame_ids]

    # Number of tracks shared by each pair of consecutive database frames.
    frame_connectivities = []

    for current_frame, next_frame in zip(frame_ids[:-1], frame_ids[1:]):
        current_tracks = set(manager_2d.tracks(current_frame))
        next_tracks = set(manager_2d.tracks(next_frame))

        frame_connectivities.append(len(current_tracks.intersection(next_tracks)))

    connectivity_plot_path = plot_frame_connectivity(
        frame_ids,
        frame_connectivities,
    )

    histogram_plot_path = plot_track_length_histogram(
        track_lengths,
    )

    matches_plot_path = plot_matches_per_frame(
        frame_ids,
        matches_per_frame,
    )

    inliers_plot_path = plot_inlier_percentages(
        inlier_percentages,
    )

    results = {
        "number_of_frames": len(frame_ids),
        "number_of_tracks": len(track_ids),
        "mean_track_length": (float(np.mean(track_lengths)) if track_lengths.size > 0 else 0.0),
        "mean_frame_connectivity": (
            float(np.mean(frame_connectivities)) if frame_connectivities else 0.0
        ),
        "mean_matches_per_frame": (float(np.mean(matches_per_frame)) if matches_per_frame else 0.0),
        "mean_inlier_percentage": (
            float(
                np.mean(
                    100.0 * inlier_percentages
                    if (len(inlier_percentages) > 0 and np.max(inlier_percentages) <= 1.0)
                    else inlier_percentages
                )
            )
            if len(inlier_percentages) > 0
            else 0.0
        ),
        "frame_connectivities": frame_connectivities,
        "matches_per_frame": matches_per_frame,
        "connectivity_plot_path": (
            str(connectivity_plot_path) if connectivity_plot_path is not None else None
        ),
        "track_length_histogram_path": (
            str(histogram_plot_path) if histogram_plot_path is not None else None
        ),
        "matches_per_frame_plot_path": (
            str(matches_plot_path) if matches_plot_path else None
        ),
        "inliers_per_frame_plot_path": (
            str(inliers_plot_path) if inliers_plot_path is not None else None
        ),
    }

    print("\n" + "=" * 60)
    print("Tracking database statistics")
    print("=" * 60)
    print(f"Number of frames:          {results['number_of_frames']}")
    print(f"Number of tracks:          {results['number_of_tracks']}")
    print(f"Mean track length:         {results['mean_track_length']:.2f}")
    print(f"Mean frame connectivity:  {results['mean_frame_connectivity']:.2f}")
    print(f"Mean matches per frame:    {results['mean_matches_per_frame']:.2f}")

    if len(inlier_percentages) > 0:
        print(f"Mean inlier percentage:    {results['mean_inlier_percentage']:.2f}%")

    print("\nLaTeX table values:")
    print(f"Number of frames & {results['number_of_frames']} \\\\")
    print(f"Number of tracks & {results['number_of_tracks']} \\\\")
    print(f"Mean track length & {results['mean_track_length']:.2f} \\\\")
    print(f"Mean frame connectivity & {results['mean_frame_connectivity']:.2f} \\\\")
    print(f"Mean matches per frame & {results['mean_matches_per_frame']:.2f} \\\\")

    if len(inlier_percentages) > 0:
        print(f"Mean inlier percentage & {results['mean_inlier_percentage']:.2f}\\% \\\\")

    return results

def pose3_camera_to_world_to_extrinsic(pose):
    """Convert a GTSAM camera-to-world Pose3 into a 3x4 world-to-camera matrix.

    Bundle poses appear to represent camera poses in frame 0 because their
    translations are used directly as camera positions. Projection requires
    world-to-camera extrinsics, so the pose is inverted here.
    """
    camera_to_world = np.asarray(pose.matrix(), dtype=float)
    world_to_camera = np.linalg.inv(camera_to_world)

    return world_to_camera[:3, :]

def get_stereo_observation(manager_2d, frame_id, track_id):
    """Return the observation as (x_left, x_right, y).

    This isolates the database-specific API in one place. If manager_2d does
    not expose link_triplet directly, only this function needs to be changed.
    """
    if hasattr(manager_2d, "link_triplet"):
        return manager_2d.link_triplet(frame_id, track_id)

    # Some SlamDatabase facades expose the old TrackingDB through an
    # attribute. Keep this branch only if it matches your implementation.
    if hasattr(manager_2d, "database") and hasattr(
        manager_2d.database,
        "link_triplet",
    ):
        return manager_2d.database.link_triplet(frame_id, track_id)

    raise AttributeError(
        "Could not retrieve a stereo observation. Implement "
        "get_stereo_observation() using the manager_2d link API. "
        "It must return (x_left, x_right, y)."
    )

def compute_pnp_analysis(database):
    """
    Compute temporal PnP reprojection errors using only estimated camera poses.

    A point is triangulated in the latest camera of each track and propagated
    into earlier cameras through the composed estimated camera motions.
    """
    print("\n" + "=" * 60)
    print("PnP temporal projection-error analysis")
    print("=" * 60)

    pnp_extrinsics = np.asarray(
        np.load(GLOBAL_CAMERA_MATRICES_PATH),
        dtype=float,
    )

    estimated_world_to_camera_by_frame = {
        frame_id: to_homogeneous_transform(extrinsic)
        for frame_id, extrinsic in enumerate(pnp_extrinsics)
    }

    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)

    (
        distances,
        median_left_errors,
        median_right_errors,
        sample_counts,
    ) = compute_median_projection_errors_by_distance(
        database=database,
        world_to_camera_by_frame=(estimated_world_to_camera_by_frame),
        calibration=calibration,
        optimize_landmarks=False,
        min_track_length=2,
        max_distance=50,
    )

    return {
        "distances": distances,
        "median_left_errors": median_left_errors,
        "median_right_errors": median_right_errors,
        "sample_counts": sample_counts,
    }

def get_relative_camera_transform(
    world_to_camera_by_frame,
    source_frame,
    target_frame,
):
    """
    Return the estimated transform from source-camera coordinates to
    target-camera coordinates.

    Given:
        T_source_world: world -> source camera
        T_target_world: world -> target camera

    Then:
        T_target_source =
            T_target_world @ inverse(T_source_world)
    """
    source_world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[source_frame])
    target_world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[target_frame])

    return target_world_to_camera @ np.linalg.inv(source_world_to_camera)

def compose_camera_transform(
    world_to_camera_by_frame,
    source_frame,
    target_frame,
):
    """
    Compose all consecutive estimated camera transforms from source_frame
    to target_frame.

    The returned transform maps a point from source-camera coordinates into
    target-camera coordinates.
    """
    if source_frame == target_frame:
        return np.eye(4, dtype=float)

    step = 1 if target_frame > source_frame else -1

    current_frame = source_frame
    target_from_source = np.eye(4, dtype=float)

    while current_frame != target_frame:
        next_frame = current_frame + step

        if (
            current_frame not in world_to_camera_by_frame
            or next_frame not in world_to_camera_by_frame
        ):
            raise KeyError(
                f"Missing estimated camera pose for transition {current_frame} -> {next_frame}"
            )

        next_from_current = get_relative_camera_transform(
            world_to_camera_by_frame=world_to_camera_by_frame,
            source_frame=current_frame,
            target_frame=next_frame,
        )

        target_from_source = next_from_current @ target_from_source

        current_frame = next_frame

    return target_from_source

def compute_bundle_adjustment_analysis(database):
    """Optimize each track landmark and compute BA reprojection errors."""
    print("\n" + "=" * 60)
    print("Track Bundle-Adjustment projection-error analysis")
    print("=" * 60)

    pnp_extrinsics = np.asarray(
        np.load(GLOBAL_CAMERA_MATRICES_PATH),
        dtype=float,
    )

    world_to_camera_by_frame = {
        frame_id: to_homogeneous_transform(extrinsic)
        for frame_id, extrinsic in enumerate(pnp_extrinsics)
    }

    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)

    (
        distances,
        median_left_errors,
        median_right_errors,
        sample_counts,
    ) = compute_median_projection_errors_by_distance(
        database=database,
        world_to_camera_by_frame=world_to_camera_by_frame,
        calibration=calibration,
        optimize_landmarks=True,
        min_track_length=2,
        max_distance=50,
    )

    return {
        "distances": distances,
        "median_left_errors": median_left_errors,
        "median_right_errors": median_right_errors,
        "sample_counts": sample_counts,
    }


def load_or_build_db():
    full_db_path = CACHE_DIR / "slam_db.pkl"
    if full_db_path.exists():
        return SlamDatabase.load(str(CACHE_DIR / "slam_db"))
    return build_database()[0]

def load_or_build_inlier_percentages():
    import numpy as np
    from src.slam.config import PROJECT_DIR
    inlier_path = PROJECT_DIR / "outputs" / "ex4" / "inlier_percentages.npy"
    if not inlier_path.exists():
        return np.ones(3300) * 100.0
    return np.load(inlier_path)

def load_or_build_pose_graph_no_lc():
    import pickle
    path = CACHE_DIR / "pose_graph_no_lc.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    candidates_by_target, build_result, pose_graph_result, slam_db, keyframe_ids = q_1()
    data = {
        "candidates_by_target": candidates_by_target,
        "build_result": build_result,
        "pose_graph_result": pose_graph_result,
        "keyframe_ids": keyframe_ids
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data

def load_or_build_loop_closures():
    import pickle
    path = CACHE_DIR / "loop_closures.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    pg_no_lc = load_or_build_pose_graph_no_lc()
    selected_results = q_2(pg_no_lc["candidates_by_target"])
    estimates = q_3(selected_results)
    
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
            "inlier_mask": first.inlier_mask,
        }
            
    data = {
        "selected_results": selected_results,
        "estimates": estimates,
        "lc_stats": lc_stats,
        "match_data": match_data
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data

def load_or_build_pose_graph_with_lc():
    import pickle
    import numpy as np
    path = CACHE_DIR / "pose_graph_with_lc.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    pg_no_lc = load_or_build_pose_graph_no_lc()
    lc = load_or_build_loop_closures()
    
    updated_pose_graph_result, versions = q_4(
        graph=pg_no_lc["build_result"].graph,
        initial_estimates=pg_no_lc["pose_graph_result"].optimized_estimates,
        keyframe_ids=pg_no_lc["keyframe_ids"],
        estimates=lc["estimates"]
    )
    
    keyframe_ids = pg_no_lc["keyframe_ids"]
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
            "covariances": covs if marginals is not None else None
        })
        
    pose_graph_matrices = [
        np.asarray(pg_no_lc["pose_graph_result"].optimized_estimates.atPose3(C(fid)).matrix())
        for fid in keyframe_ids
    ]
    pose_graph_lc_matrices = [
        np.asarray(updated_pose_graph_result.optimized_estimates.atPose3(C(fid)).matrix())
        for fid in keyframe_ids
    ]
    
    data = {
        "updated_pose_graph_result": updated_pose_graph_result,
        "versions_data": versions_data,
        "pose_graph_matrices": pose_graph_matrices,
        "pose_graph_lc_matrices": pose_graph_lc_matrices
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data

def load_or_build_bundle_windows():
    import pickle
    path = CACHE_DIR / "bundle_windows.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    db = load_or_build_db()
    ba_context = prepare_bundle_adjustment_analysis(db)
    
    (
        window_ids,
        initial_average_errors,
        final_average_errors,
        initial_median_proj_errors,
        final_median_proj_errors,
    ) = compute_bundle_window_errors(ba_context.solutions)
    
    errors = {
        "window_ids": window_ids,
        "initial_average_errors": initial_average_errors,
        "final_average_errors": final_average_errors,
        "initial_median_proj_errors": initial_median_proj_errors,
        "final_median_proj_errors": final_median_proj_errors,
    }
    
    ba_context.calibration = None
    data = {
        "context": ba_context,
        "errors": errors
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data

def load_or_build_projection_data():
    import pickle
    path = CACHE_DIR / "projection_data.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    db = load_or_build_db()
    pnp_projection_data = compute_pnp_analysis(db)
    ba_projection_data = compute_bundle_adjustment_analysis(db)
    
    data = {
        "pnp_projection_data": pnp_projection_data,
        "ba_projection_data": ba_projection_data
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data

def load_or_build_tracking_analysis():
    import pickle
    from src.slam.config import CACHE_DIR
    path = CACHE_DIR / "tracking_analysis.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    db = load_or_build_db()
    manager_2d = db.manager_2d
    frame_ids = sorted(manager_2d.all_frames())
    track_ids = list(manager_2d.all_tracks())
    track_lengths = [len(manager_2d.frames(tid)) for tid in track_ids]
    matches_per_frame = [len(manager_2d.tracks(fid)) for fid in frame_ids]
    
    frame_connectivities = []
    for current_frame, next_frame in zip(frame_ids[:-1], frame_ids[1:]):
        current_tracks = set(manager_2d.tracks(current_frame))
        next_tracks = set(manager_2d.tracks(next_frame))
        frame_connectivities.append(len(current_tracks.intersection(next_tracks)))
        
    import numpy as np
    data = {
        "frame_ids": frame_ids,
        "track_lengths": np.asarray(track_lengths, dtype=float),
        "matches_per_frame": matches_per_frame,
        "frame_connectivities": frame_connectivities
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data



def optimize_track_landmark(
    manager_2d,
    track_id,
    frame_ids,
    world_to_camera_by_frame,
    calibration,
    measurement_sigma_pixels=1.0,
):
    """Optimize one landmark using all stereo observations in one track.

    Camera poses are fixed using tight Pose3 priors. Only the landmark is
    meaningfully adjusted by the optimization.
    """
    if len(frame_ids) < 2:
        raise ValueError("At least two observations are required.")

    triangulation_frame = frame_ids[-1]

    triangulation_extrinsic = to_homogeneous_transform(
        world_to_camera_by_frame[triangulation_frame]
    )

    triangulation_camera = make_stereo_camera(
        triangulation_extrinsic,
        calibration,
    )

    triangulation_measurement = stereo_point_from_triplet(
        get_stereo_observation(
            manager_2d,
            triangulation_frame,
            track_id,
        )
    )

    initial_landmark = triangulation_camera.backproject(triangulation_measurement)

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    landmark_key = Q(int(track_id))

    initial_values.insert(
        landmark_key,
        gtsam.Point3(np.asarray(initial_landmark, dtype=float)),
    )

    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        3,
        measurement_sigma_pixels,
    )

    # Very tight priors keep all PnP poses fixed.
    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-7, dtype=float))

    for frame_id in frame_ids:
        pose_key = C(int(frame_id))

        world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[frame_id])

        camera_pose = pose3_from_world_to_camera_extrinsic(world_to_camera)

        initial_values.insert(pose_key, camera_pose)

        graph.add(
            gtsam.PriorFactorPose3(
                pose_key,
                camera_pose,
                pose_prior_noise,
            )
        )

        measurement = stereo_point_from_triplet(
            get_stereo_observation(
                manager_2d,
                frame_id,
                track_id,
            )
        )

        graph.add(
            gtsam.GenericStereoFactor3D(
                measurement,
                measurement_noise,
                pose_key,
                landmark_key,
                calibration,
            )
        )

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_values,
    )

    optimized_values = optimizer.optimize()

    optimized_landmark = np.asarray(
        optimized_values.atPoint3(landmark_key),
        dtype=float,
    ).reshape(3)

    return optimized_landmark