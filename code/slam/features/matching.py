import cv2
import numpy as np

from typing import Sequence, List, Tuple

from .detectors import extract_features, FeatureType


KNN_NEIGHBORS = 2
RATIO_THRESHOLD = 0.75


def create_matcher(feature_type: FeatureType = "sift"):
    """Create a brute-force matcher compatible with SIFT or ORB descriptors."""
    feature_type = feature_type.lower()

    if feature_type == "sift":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    if feature_type == "orb":
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    raise ValueError(f"Unsupported feature_type: {feature_type}")

def match_features(
    des1: np.ndarray,
    des2: np.ndarray,
    feature_type: FeatureType = "sift",
    k: int = KNN_NEIGHBORS,
) -> Sequence[Sequence[cv2.DMatch]]:
    """Return KNN descriptor matches from image 1 descriptors to image 2 descriptors."""
    matcher = create_matcher(feature_type)
    return matcher.knnMatch(des1, des2, k=k)


def filter_matches_ratio(
    knn_matches: Sequence[Sequence[cv2.DMatch]],
    ratio: float = RATIO_THRESHOLD,
) -> Tuple[List[cv2.DMatch], List[cv2.DMatch]]:
    """Filters KNN matches using Lowe's ratio test."""
    good_matches = []
    rejected_matches = []

    for neighbors in knn_matches:
        if len(neighbors) < 2:
            continue

        m, n = neighbors

        if m.distance < ratio * n.distance:
            good_matches.append(m)
        else:
            rejected_matches.append(m)

    return good_matches, rejected_matches


def match_and_filter(
    des1: np.ndarray,
    des2: np.ndarray,
    feature_type: FeatureType = "sift",
    ratio: float = RATIO_THRESHOLD,
) -> List[cv2.DMatch]:
    """Match descriptors and return only matches passing Lowe's ratio test."""
    knn_matches = match_features(des1, des2, feature_type=feature_type)
    good_matches, _ = filter_matches_ratio(knn_matches, ratio)
    return good_matches


def get_matched_points(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert DMatch objects into two aligned Nx2 arrays of pixel locations."""
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

    return pts1, pts2


def extract_and_match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    feature_type: FeatureType = "sift",
    num_features: int | None = None,
    ratio_threshold: float = RATIO_THRESHOLD,
) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
    """Extract SIFT/ORB features from two images and return ratio-filtered matches."""
    kp1, des1 = extract_features(img1, feature_type=feature_type, num_features=num_features)
    kp2, des2 = extract_features(img2, feature_type=feature_type, num_features=num_features)

    knn_matches = match_features(des1, des2, feature_type=feature_type)
    good_matches, _ = filter_matches_ratio(knn_matches, ratio_threshold)

    return kp1, kp2, good_matches
