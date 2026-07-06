import numpy as np

from src.slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic
from src.slam.pipeline.pose_graph_pipeline import (
    load_ex6_inputs,
    solve_bundle_windows_and_extract_constraints,
)
from src.slam.loop_closure.candidate_detection import (
    ConsensusMatchResult,
    ConsensusMatcher,
    LoopClosureCandidate,
    RelativePoseEstimate,
    detect_candidates_for_keyframe,
    refine_relative_pose_with_bundle_adjustment,
    score_candidates_for_keyframe,
    select_spread_loop_closures,
)
from src.slam.pose_graph.graph_builder import build_pose_graph
from src.slam.pose_graph.optimization import optimize_pose_graph

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import C

from src.slam.config import SEQUENCE_DIR



MAHALANOBIS_THRESHOLD = 313_337.7

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

    return (
        translation_m <= max_translation_m
        and rotation_deg <= max_rotation_deg
    )


def _print_mahalanobis_threshold_sweep(
    all_candidates: list,
    *,
    max_translation_m: float = 5.0,
    max_rotation_deg: float = 20.0,
) -> float:
    """
    Pick a Mahalanobis threshold empirically.

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
        f"{'threshold':>15} | {'close recall':>12} | "
        f"{'selected':>10} | {'selected / target':>18}"
    )
    print("-" * 67)

    chosen_threshold = None

    for percentile in (50, 75, 90, 95, 97.5, 99):
        threshold = float(np.percentile(close_scores, percentile))

        selected = [
            candidate
            for candidate in all_candidates
            if candidate.mahalanobis_squared <= threshold
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

        targets_with_selected = len(
            {candidate.target_frame for candidate in selected}
        )

        selected_per_target = (
            len(selected) / targets_with_selected
            if targets_with_selected > 0
            else 0.0
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
    """
    Return the temporally nonlocal candidate with the smallest Euclidean
    relative translation, regardless of Mahalanobis score.
    """
    all_candidates = [
        candidate
        for candidates in candidates_by_target.values()
        for candidate in candidates
    ]

    if not all_candidates:
        raise RuntimeError("No temporally nonlocal candidate pairs were found.")

    return min(
        all_candidates,
        key=lambda candidate: np.linalg.norm(candidate.relative_delta[3:]),
    )


def _load_left_image(frame_id: int) -> np.ndarray:
    """
    Load KITTI sequence-00 left image for a raw frame index.

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
        [
            optimized_values.atPose3(C(frame_id)).translation()
            for frame_id in keyframe_ids
        ],
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

    axes[2].set_title(
        "Optimized trajectory\n"
        f"{translation_m:.2f} m, {rotation_deg:.2f}°"
    )
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("z [m]")
    axes[2].axis("equal")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    plt.show()


def _print_candidate_debug(candidate) -> None:
    """
    Print the relative motion, accumulated path uncertainty, and rough
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

    print(
        f"\n    c_{candidate.source_frame} -> "
        f"c_{candidate.target_frame}"
    )
    print(f"      d^2:                {candidate.mahalanobis_squared:.3f}")
    print(f"      path edges:         {len(candidate.path_frame_ids) - 1}")
    print(f"      path:               {candidate.path_frame_ids}")
    print(f"      rotation magnitude: {rotation_magnitude_deg:.3f} deg")
    print(f"      translation norm:   {translation_magnitude:.3f} m")
    print(
        "      delta [rad, m]:     "
        f"{np.array2string(delta, precision=6)}"
    )
    print(
        "      path std [rad, m]:  "
        f"{np.array2string(standard_deviations, precision=6)}"
    )
    print(
        "      cov eig:            "
        f"{np.array2string(covariance_eigenvalues, precision=3)}"
    )
    print(
        "      rough d^2 terms:    "
        f"{np.array2string(diagonal_contributions, precision=1)}"
    )


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
        target_candidates.sort(
            key=lambda candidate: candidate.mahalanobis_squared
        )

    return candidates_by_target

def q_1() -> dict[int, list[LoopClosureCandidate]]:
    """
    Section 7.1: detect loop-closure candidates using an empirically
    calibrated Mahalanobis threshold.

    The threshold was selected offline as the smallest value that retained
    at least 90% of temporally nonlocal pairs within 5 m and 20 degrees,
    while keeping the number of consensus-matching attempts small.
    """
    min_keyframe_separation = 5
    keyframe_step = 10

    db, world_to_camera_extrinsics, calibration = load_ex6_inputs()

    _, constraints = solve_bundle_windows_and_extract_constraints(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
        keyframe_step=keyframe_step,
        verbose=True,
    )

    if not constraints:
        raise RuntimeError("No relative-pose constraints were extracted.")

    first_frame = constraints[0].start_frame
    first_pose = pose3_from_world_to_camera_extrinsic(
        world_to_camera_extrinsics[first_frame]
    )

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
    print(f"Keyframes tested:             {len(keyframe_ids)}")
    print(f"Keyframe interval:            {keyframe_step} frames")
    print(f"Minimum temporal separation:  {min_keyframe_separation}")
    print(f"Mahalanobis threshold:        {MAHALANOBIS_THRESHOLD:.1f}")
    print(
        "Threshold rationale:          smallest empirical threshold "
        "with at least 90% proxy-close-pair recall"
    )

    if len(keyframe_ids) != 310:
        print(
            "Warning: expected 310 keyframes according to the exercise, "
            f"but built a graph with {len(keyframe_ids)}."
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

    accepted_candidates.sort(
        key=lambda candidate: candidate.mahalanobis_squared
    )

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

    return candidates_by_target

def q_2(
    candidates_by_target: dict[int, list[LoopClosureCandidate]],
) -> list[ConsensusMatchResult]:
    """
    Section 7.2: verify Mahalanobis-selected candidates using four-view
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

    successful_results = [
        result
        for result in results
        if result.success
    ]

    print("\n" + "=" * 60)
    print("Consensus-matching summary")
    print("=" * 60)
    print(f"Candidates tested:      {len(results)}")
    print(f"Verified loop closures: {len(successful_results)}")

    if successful_results:
        print("\nAccepted loop closures:")
        for result in successful_results:
            candidate = result.candidate
            print(
                f"  c_{candidate.source_frame} -> c_{candidate.target_frame}: "
                f"{result.num_inliers} inliers "
                f"({100.0 * result.inlier_ratio:.1f}%)"
            )

    return successful_results


def q_3(
    verified_results: list[ConsensusMatchResult],
) -> list[RelativePoseEstimate]:
    """
    Section 7.3: refine selected consensus matches using two-frame stereo BA
    and extract a relative-pose covariance for each loop constraint.
    """
    representatives = select_spread_loop_closures(
        verified_results,
        num_representatives=3,
    )

    print("\n" + "=" * 60)
    print("[7.3] Relative Pose Estimation")
    print("=" * 60)
    print(f"Verified loop closures:        {len(verified_results)}")
    print(f"Representative links refined:  {len(representatives)}")
    print(
        "Initialization:                source pose fixed at identity; "
        "target initialized from PnP/RANSAC"
    )
    print(
        "Covariance extraction:         "
        "marginal covariance of target pose after two-frame stereo BA"
    )

    estimates: list[RelativePoseEstimate] = []

    for index, result in enumerate(representatives, start=1):
        candidate = result.candidate

        print(
            f"\nLoop {index}/{len(representatives)}: "
            f"c_{candidate.source_frame} -> c_{candidate.target_frame}"
        )

        estimate = refine_relative_pose_with_bundle_adjustment(result)
        estimates.append(estimate)

        covariance_std = np.sqrt(np.diag(estimate.covariance))

        print(f"  Inlier landmarks:       {estimate.num_landmarks}")
        print(
            f"  BA error:               "
            f"{estimate.initial_error:.2f} -> {estimate.final_error:.2f}"
        )
        print(
            "  Relative-pose std:       "
            f"{np.array2string(covariance_std, precision=5)}"
        )
        print(
            f"  Covariance condition:    "
            f"{np.linalg.cond(estimate.covariance):.2e}"
        )

    return estimates

def main() -> None:
    candidates_by_target = q_1()
    verified_results = q_2(candidates_by_target)
    q_3(verified_results)


if __name__ == "__main__":
    main()
