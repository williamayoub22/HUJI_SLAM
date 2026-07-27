"""
Provides diagnostics components and utilities for the SLAM pipeline.
"""

from pathlib import Path

import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import C

from src.slam import config
from src.slam.loop_closure.candidates import LoopClosureCandidate


def _print_mahalanobis_threshold_sweep(
    all_candidates: list, *, max_translation_m: float = 5.0, max_rotation_deg: float = 20.0
) -> float:
    """Pick a Mahalanobis threshold empirically.

    'Close' is used only as an offline proxy for a potential visual overlap.
    The actual algorithm later uses only the Mahalanobis threshold followed
    by consensus matching.
    """
    if not all_candidates:
        raise RuntimeError("No candidates available for threshold calibration.")
    scores = np.asarray(
        [candidate.mahalanobis_squared for candidate in all_candidates], dtype=float
    )
    close_candidates = [
        candidate for candidate in all_candidates if _is_geometrically_close(candidate)
    ]
    if not close_candidates:
        raise RuntimeError(
            "No geometrically close nonlocal pairs found. Increase the calibration radii or inspect the trajectory."
        )
    close_scores = np.asarray(
        [candidate.mahalanobis_squared for candidate in close_candidates], dtype=float
    )
    print("\n" + "=" * 60)
    print("Empirical Mahalanobis-threshold calibration")
    print("=" * 60)
    print(
        f"Calibration close-pair proxy: translation <= {max_translation_m:.1f} m, rotation <= {max_rotation_deg:.1f} deg"
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
        selected_close = [candidate for candidate in selected if _is_geometrically_close(candidate)]
        recall = len(selected_close) / len(close_candidates)
        targets_with_selected = len({candidate.target_frame for candidate in selected})
        selected_per_target = (
            len(selected) / targets_with_selected if targets_with_selected > 0 else 0.0
        )
        print(
            f"{threshold:15.1f} | {recall:11.1%} | {len(selected):10d} | {selected_per_target:18.2f}"
        )
        if chosen_threshold is None and recall >= 0.9:
            chosen_threshold = threshold
    if chosen_threshold is None:
        chosen_threshold = float(np.max(close_scores))
    print("\nRecommended initial threshold:")
    print(f"  mahalanobis_threshold = {chosen_threshold:.1f}")
    return chosen_threshold


def _is_geometrically_close(candidate) -> bool:
    """Offline calibration label only; never used as the actual online gate."""
    delta = np.asarray(candidate.relative_delta, dtype=float)
    translation_m = float(np.linalg.norm(delta[3:]))
    rotation_deg = float(np.degrees(np.linalg.norm(delta[:3])))
    return translation_m <= config.MAX_TRANSLATION_M and rotation_deg <= config.MAX_ROTATION_DEG


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
    positions = np.array(
        [
            optimized_values.atPose3(C(frame_id)).translation()
            for frame_id in keyframe_ids
            if optimized_values.exists(C(frame_id))
        ]
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
    axes[2].scatter([source_position[0]], [source_position[2]], s=100, label=f"c_{source_frame}")
    axes[2].scatter([target_position[0]], [target_position[2]], s=100, label=f"c_{target_frame}")
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


def _closest_spatial_candidate(candidates_by_target: dict[int, list]):
    """Return the temporally nonlocal candidate with the smallest Euclidean
    relative translation, regardless of Mahalanobis score.
    """
    all_candidates = [
        candidate for candidates in candidates_by_target.values() for candidate in candidates
    ]
    if not all_candidates:
        raise RuntimeError("No temporally nonlocal candidate pairs were found.")
    return min(all_candidates, key=lambda candidate: np.linalg.norm(candidate.relative_delta[3:]))


def _print_candidate_debug(candidate) -> None:
    """Print the relative motion, accumulated path uncertainty, and rough
    per-coordinate Mahalanobis contributions for one candidate.
    """
    delta = np.asarray(candidate.relative_delta, dtype=float)
    covariance = np.asarray(candidate.covariance, dtype=float)
    standard_deviations = np.sqrt(np.diag(covariance))
    rotation_vector = delta[:3]
    translation_vector = delta[3:]
    rotation_magnitude_deg = np.degrees(np.linalg.norm(rotation_vector))
    translation_magnitude = np.linalg.norm(translation_vector)
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


def _group_candidates_by_target(candidates: list) -> dict[int, list]:
    """Group already-scored candidates by their later keyframe."""
    candidates_by_target: dict[int, list] = {}
    for candidate in candidates:
        candidates_by_target.setdefault(candidate.target_frame, []).append(candidate)
    for target_candidates in candidates_by_target.values():
        target_candidates.sort(key=lambda candidate: candidate.mahalanobis_squared)
    return candidates_by_target


def _load_left_image(frame_id: int) -> np.ndarray:
    """Load KITTI sequence-00 left image for a raw frame index.

    Adjust image_0 if your project uses another left-image directory.
    """
    image_path = Path(config.SEQUENCE_DIR) / "image_0" / f"{frame_id:06d}.png"
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
