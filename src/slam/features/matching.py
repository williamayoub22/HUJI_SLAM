"""Descriptor matching and match-filtering utilities."""

from collections.abc import Sequence

import cv2
import numpy as np

from .detectors import FeatureType, extract_features, normalize_feature_type

KNN_NEIGHBORS = 2
from .. import config


def create_matcher(
    feature_type: FeatureType = "sift",
) -> cv2.BFMatcher:
    """Create a brute-force matcher compatible with a descriptor type.

    Args:
        feature_type: Feature detector that produced the descriptors.

    Returns:
        Brute-force matcher using L2 for SIFT and Hamming distance for binary
        descriptors.
    """
    normalized_type = normalize_feature_type(feature_type)

    if normalized_type == "sift":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def match_features(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    feature_type: FeatureType = "sift",
    k: int = KNN_NEIGHBORS,
) -> Sequence[Sequence[cv2.DMatch]]:
    """Return the ``k`` nearest descriptor matches from image 1 to image 2.

    Args:
        descriptors1: Descriptor matrix for the first image.
        descriptors2: Descriptor matrix for the second image.
        feature_type: Feature detector that produced the descriptors.
        k: Number of nearest neighbors returned per query descriptor.

    Returns:
        K-nearest-neighbor matches for every descriptor in ``descriptors1``.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    matcher = create_matcher(feature_type)
    return matcher.knnMatch(descriptors1, descriptors2, k=k)


def filter_matches_ratio(
    knn_matches: Sequence[Sequence[cv2.DMatch]],
    ratio: float = config.RATIO_THRESHOLD,
) -> tuple[list[cv2.DMatch], list[cv2.DMatch]]:
    """Filter two-neighbor matches using Lowe's ratio test.

    Args:
        knn_matches: KNN matches with at least two candidates per query.
        ratio: Maximum accepted ratio between best and second-best distances.

    Returns:
        Matches that pass the ratio test and rejected best-match candidates.

    Raises:
        ValueError: If ``ratio`` is not in the interval ``(0, 1]``.
    """
    if not 0 < ratio <= 1:
        raise ValueError("ratio must lie in the interval (0, 1].")

    accepted_matches: list[cv2.DMatch] = []
    rejected_matches: list[cv2.DMatch] = []

    for neighbors in knn_matches:
        if len(neighbors) < 2:
            continue

        best_match, second_best_match = neighbors

        if best_match.distance < ratio * second_best_match.distance:
            accepted_matches.append(best_match)
        else:
            rejected_matches.append(best_match)

    return accepted_matches, rejected_matches


def match_and_filter(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    feature_type: FeatureType = "sift",
    ratio: float = config.RATIO_THRESHOLD,
) -> list[cv2.DMatch]:
    """Match descriptors and retain only matches passing Lowe's ratio test."""
    knn_matches = match_features(
        descriptors1,
        descriptors2,
        feature_type=feature_type,
        k=KNN_NEIGHBORS,
    )

    accepted_matches, _ = filter_matches_ratio(knn_matches, ratio)
    return accepted_matches


def get_matched_points(
    keypoints1: Sequence[cv2.KeyPoint],
    keypoints2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert descriptor matches into aligned ``(N, 2)`` pixel-coordinate arrays.

    Args:
        keypoints1: Keypoints detected in the first image.
        keypoints2: Keypoints detected in the second image.
        matches: Correspondences between the two keypoint sets.

    Returns:
        Pixel coordinates from the first and second images in match order.
    """
    if not matches:
        empty_points = np.empty((0, 2), dtype=np.float32)
        return empty_points, empty_points.copy()

    points1 = np.asarray(
        [keypoints1[match.queryIdx].pt for match in matches],
        dtype=np.float32,
    )
    points2 = np.asarray(
        [keypoints2[match.trainIdx].pt for match in matches],
        dtype=np.float32,
    )

    return points1, points2


def extract_and_match_features(
    image1: np.ndarray,
    image2: np.ndarray,
    feature_type: FeatureType = "sift",
    num_features: int | None = None,
    ratio_threshold: float = config.STEREO_RATIO_THRESHOLD,
    use_ratio_test: bool = False,
) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], np.ndarray, list[cv2.DMatch]]:
    """Extract features from two images and compute descriptor correspondences.

    When ``use_ratio_test`` is false, this returns the nearest descriptor match
    for every query descriptor. When true, it retains only matches passing
    Lowe's ratio test.

    Args:
        image1: First input image.
        image2: Second input image.
        feature_type: Detector and descriptor implementation to use.
        num_features: Requested maximum number of SIFT or ORB features.
        ratio_threshold: Lowe ratio-test threshold.
        use_ratio_test: Whether to reject ambiguous nearest-neighbor matches.

    Returns:
        Keypoints from both images, descriptors from the first image, and descriptor matches between them.
    """
    keypoints1, descriptors1 = extract_features(
        image1,
        feature_type=feature_type,
        num_features=num_features,
    )
    keypoints2, descriptors2 = extract_features(
        image2,
        feature_type=feature_type,
        num_features=num_features,
    )

    knn_matches = match_features(
        descriptors1,
        descriptors2,
        feature_type=feature_type,
        k=KNN_NEIGHBORS,
    )

    if use_ratio_test:
        accepted_matches, _ = filter_matches_ratio(
            knn_matches,
            ratio=ratio_threshold,
        )
        return keypoints1, keypoints2, descriptors1, accepted_matches

    nearest_matches = [neighbors[0] for neighbors in knn_matches if neighbors]

    return keypoints1, keypoints2, descriptors1, nearest_matches
