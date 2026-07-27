"""Robust PnP estimation with four-view reprojection-consistency RANSAC."""

import math

import cv2
import numpy as np

from .projection import solve_pnp_safe
from .projection import count_supporters
from .. import config




def _required_ransac_iterations(
    inlier_ratio: float,
    confidence: float,
    sample_size: int,
) -> int:
    """Return the number of RANSAC iterations needed for a target confidence.

    Args:
        inlier_ratio: Estimated probability that one correspondence is an inlier.
        confidence: Desired probability of sampling at least one all-inlier set.
        sample_size: Number of correspondences in one minimal sample.

    Returns:
        Required number of iterations. Returns one when an all-inlier sample is
        effectively guaranteed.
    """
    probability_good_sample = inlier_ratio**sample_size

    if probability_good_sample >= 1.0:
        return 1

    probability_bad_sample = 1.0 - probability_good_sample
    probability_bad_sample = np.clip(
        probability_bad_sample,
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
    )

    return int(math.ceil(math.log(1.0 - confidence) / math.log(probability_bad_sample)))


def ransac_pnp(
    points_3d: np.ndarray,
    left1: np.ndarray,
    left0: np.ndarray,
    right0: np.ndarray,
    right1: np.ndarray,
    intrinsic_matrix: np.ndarray,
    left_projection_matrix: np.ndarray,
    right_projection_matrix: np.ndarray,
    max_iterations: int = config.MAX_RANSAC_ITERATIONS,
    confidence: float = config.RANSAC_CONFIDENCE,
    sample_size: int = config.PNP_SAMPLE_SIZE,
    supporter_threshold_pixels: float = config.SUPPORTER_THRESHOLD_PIXELS,
    max_translation_m: float = config.MAX_TRANSLATION_PNP,
    min_inliers: int = config.MIN_INLIERS_PNP,
    seed: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate a left0-to-left1 transform using RANSAC with PnP.

    Candidate transforms are evaluated using reprojection consistency in all
    four images: left0, right0, left1, and right1. The best model is refined
    with iterative PnP using all of its supporters.

    Args:
        points_3d: 3D points expressed in left-camera-0 coordinates, shape
            ``(N, 3)``.
        left1: Corresponding observations in left image 1, shape ``(N, 2)``.
        left0: Observations in left image 0, shape ``(N, 2)``.
        right0: Observations in right image 0, shape ``(N, 2)``.
        right1: Observations in right image 1, shape ``(N, 2)``.
        intrinsic_matrix: Left-camera intrinsic matrix, shape ``(3, 3)``.
        left_projection_matrix: Left-camera-0 projection matrix, shape
            ``(3, 4)``.
        right_projection_matrix: Right-camera-0 projection matrix, shape
            ``(3, 4)``.
        max_translation: Maximum allowed translation norm for a candidate pose.
        min_inliers: Minimum supporters required for a valid result.
        seed: Optional seed for reproducible random sampling.

    Returns:
        The refined ``[R | t]`` transform from left0 to left1 and its supporter
        mask. Returns ``(None, None)`` when no valid model is found.

    Raises:
        ValueError: If parameters are invalid or correspondence arrays are
            misaligned.
    """
    num_points = len(points_3d)

    if num_points < config.PNP_SAMPLE_SIZE:
        return None, None

    if not 0.0 < config.RANSAC_CONFIDENCE < 1.0:
        raise ValueError("confidence must lie in the interval (0, 1).")

    if config.MAX_RANSAC_ITERATIONS <= 0:
        raise ValueError("max_iterations must be positive.")

    if max_translation_m <= 0:
        raise ValueError("max_translation_m must be positive.")

    if min_inliers < config.PNP_SAMPLE_SIZE:
        raise ValueError(f"min_inliers must be at least {config.PNP_SAMPLE_SIZE}.")

    correspondence_arrays = (left0, right0, left1, right1)
    if any(len(points) != num_points for points in correspondence_arrays):
        raise ValueError("points_3d and all four image-point arrays must have equal length.")

    rng = np.random.default_rng(seed)

    best_transform: np.ndarray | None = None
    best_mask: np.ndarray | None = None
    max_supporters = 0

    current_max_iterations = config.MAX_RANSAC_ITERATIONS
    iteration = 0

    while iteration < current_max_iterations:
        sample_indices = rng.choice(
            num_points,
            size=config.PNP_SAMPLE_SIZE,
            replace=False,
        )

        candidate_transform = solve_pnp_safe(
            points_3d[sample_indices],
            left1[sample_indices],
            intrinsic_matrix,
            cv2.SOLVEPNP_EPNP,
        )

        if candidate_transform is not None:
            translation_norm = np.linalg.norm(candidate_transform[:3, 3])

            if translation_norm < max_translation_m:
                try:
                    supporter_mask, _ = count_supporters(
                        transform_left0_to_left1=candidate_transform,
                        points_3d_left0=points_3d,
                        left0=left0,
                        right0=right0,
                        left1=left1,
                        right1=right1,
                        intrinsic_matrix=intrinsic_matrix,
                        left_projection_matrix=left_projection_matrix,
                        right_projection_matrix=right_projection_matrix,
                        threshold_pixels=supporter_threshold_pixels,
                    )
                except ValueError:
                    supporter_mask = np.zeros(len(points_3d), dtype=bool)

                supporter_count = int(np.sum(supporter_mask))

                if supporter_count > max_supporters:
                    best_transform = candidate_transform
                    best_mask = supporter_mask
                    max_supporters = supporter_count

                    inlier_ratio = supporter_count / num_points
                    required_iterations = _required_ransac_iterations(
                        inlier_ratio=inlier_ratio,
                        confidence=config.RANSAC_CONFIDENCE,
                        sample_size=config.PNP_SAMPLE_SIZE,
                    )

                    current_max_iterations = min(
                        current_max_iterations,
                        max(iteration + 1, required_iterations),
                    )

        iteration += 1

    if best_transform is None or best_mask is None:
        return None, None

    if max_supporters < min_inliers:
        return None, None

    refined_transform = solve_pnp_safe(
        points_3d[best_mask],
        left1[best_mask],
        intrinsic_matrix,
        cv2.SOLVEPNP_ITERATIVE,
    )

    if refined_transform is not None:
        translation_norm = np.linalg.norm(refined_transform[:3, 3])

        if translation_norm < max_translation_m:
            refined_mask, _ = count_supporters(
                transform_left0_to_left1=refined_transform,
                points_3d_left0=points_3d,
                left0=left0,
                right0=right0,
                left1=left1,
                right1=right1,
                intrinsic_matrix=intrinsic_matrix,
                left_projection_matrix=left_projection_matrix,
                right_projection_matrix=right_projection_matrix,
                threshold_pixels=supporter_threshold_pixels,
            )

            refined_supporter_count = int(np.sum(refined_mask))

            if refined_supporter_count >= min_inliers:
                best_transform = refined_transform
                best_mask = refined_mask
                max_supporters = refined_supporter_count

    if max_supporters < min_inliers:
        return None, None

    return best_transform, best_mask
