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
)
from src.slam.ba.optimization import _validate_graph_keys
from src.slam.ba.results import BundleAdjustmentResult, ProjectionFactorMetadata, BundleWindowSolution
from src.slam.ba.window_selection import collect_window_tracks, bundle_windows_from_keyframes, choose_keyframes_by_motion
from src.slam.config import (
    SEQUENCE_DIR,
    EX8_OUTPUT_DIR,
    GT_POSES_PATH
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
    EX8_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    P1, P2 = read_stereo_calibration()
    K = P1[:, :3]
    calibration = gtsam.Cal3_S2Stereo(P1[0,0], P1[1,1], 0.0, P1[0,2], P1[1,2], -P2[0,3] / P1[0,0])

    db_path = EX8_OUTPUT_DIR / "slam_db"
    full_db_path = EX8_OUTPUT_DIR / "slam_db.pkl"

    if full_db_path.exists():
        print(f"Loading database from {full_db_path}...")
        slam_db = SlamDatabase.load(str(db_path))
        return slam_db, calibration

    print("Building database from scratch...")
    
    slam_db = SlamDatabase()
    
    total_frames = len(list((SEQUENCE_DIR / "image_0").glob("*.png")))
    
    global_T = np.eye(4)
    slam_db.manager_poses.add_pose(0, global_T)
    
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

    db_path = EX8_OUTPUT_DIR / "slam_db"
    slam_db.serialize(str(db_path))
    
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


def q_5(
    verified_results: list[ConsensusMatchResult],
    pose_graph_result,
    updated_pose_graph_result,
    versions,
    slam_db,
    keyframe_ids,
):
    print("\n" + "=" * 60)
    print("[7.5] Plotting and Questions")
    print("=" * 60)

    num_successful = sum(1 for r in verified_results if r.success)
    print(f"Number of successful loop closures detected: {num_successful}")

    EX8_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = EX8_OUTPUT_DIR

    if num_successful > 0:
        successful = [r for r in verified_results if r.success]
        plot_consensus_match(successful[0], output_dir / "consensus_match.png")
        print("Generated consensus_match.png")

    plot_pose_graphs_versions(versions, keyframe_ids, output_dir / "pose_graph_versions.png")
    print("Generated pose_graph_versions.png")

    gt_world_to_camera_extrinsics = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)

    plot_pose_graph_comparisons(
        pose_graph_result.optimized_estimates,
        updated_pose_graph_result.optimized_estimates,
        gt_world_to_camera_extrinsics,
        keyframe_ids,
        output_dir / "pose_graph_comparison.png",
    )
    print("Generated pose_graph_comparison.png")

    plot_absolute_location_error(
        pose_graph_result.optimized_estimates,
        updated_pose_graph_result.optimized_estimates,
        gt_world_to_camera_extrinsics,
        keyframe_ids,
        output_dir / "absolute_location_error.png",
    )
    print("Generated absolute_location_error.png")

    marginals_no_lc = versions[0][2]
    marginals_lc = versions[-1][2]

    plot_location_uncertainty(
        marginals_no_lc, marginals_lc, keyframe_ids, output_dir / "location_uncertainty.png"
    )
    print("Generated location_uncertainty.png (Measure: det(Cov_translation))")

    def compute_location_errors(values, gt_world_to_camera_extrinsics, keyframe_ids):
        errors = []

        for frame_id in keyframe_ids:
            estimated_position = np.asarray(values.atPose3(C(frame_id)).translation())

            gt_pose = pose3_from_world_to_camera_extrinsic(gt_world_to_camera_extrinsics[frame_id])
            gt_position = np.asarray(gt_pose.translation())

            errors.append(np.linalg.norm(estimated_position - gt_position))

        return np.asarray(errors)

    errors_without_lc = compute_location_errors(
        pose_graph_result.optimized_estimates,
        gt_world_to_camera_extrinsics,
        keyframe_ids,
    )

    errors_with_lc = compute_location_errors(
        updated_pose_graph_result.optimized_estimates,
        gt_world_to_camera_extrinsics,
        keyframe_ids,
    )

    print("\nAbsolute location error against GT")
    print("----------------------------------")
    print(
        f"Without LC: max={errors_without_lc.max():.2f} m, "
        f"mean={errors_without_lc.mean():.2f} m, "
        f"final={errors_without_lc[-1]:.2f} m"
    )
    print(
        f"With LC:    max={errors_with_lc.max():.2f} m, "
        f"mean={errors_with_lc.mean():.2f} m, "
        f"final={errors_with_lc[-1]:.2f} m"
    )


def main() -> None:
    (
        candidates_by_target,
        build_result,
        pose_graph_result,
        slam_db,
        keyframe_ids,
    ) = q_1()
    selected_results = q_2(candidates_by_target)
    estimates = q_3(selected_results)

    updated_pose_graph_result, versions = q_4(
        graph=build_result.graph,
        initial_estimates=pose_graph_result.optimized_estimates,
        keyframe_ids=keyframe_ids,
        estimates=estimates,
    )

    q_5(
        verified_results=selected_results,
        pose_graph_result=pose_graph_result,
        updated_pose_graph_result=updated_pose_graph_result,
        versions=versions,
        slam_db=slam_db,
        keyframe_ids=keyframe_ids,
    )

if __name__ == "__main__":
    main()
