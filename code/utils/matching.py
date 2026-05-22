import cv2
import numpy as np

from typing import Sequence, List, Tuple

from utils.features import extract_features


KNN_NEIGHBORS = 2
RATIO_THRESHOLD = 0.75


def match_features(
    des1: np.ndarray,
    des2: np.ndarray,
    k: int = KNN_NEIGHBORS,
) -> Sequence[Sequence[cv2.DMatch]]:
    """
    Returns KNN descriptor matches.

    For every descriptor in image 1, returns its k nearest neighbors in image 2.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return bf.knnMatch(des1, des2, k=k)


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
    ratio: float = RATIO_THRESHOLD,
) -> List[cv2.DMatch]:
    """
    Matches descriptors and applies Lowe's ratio test in one step.
    Returns only the good matches.
    """
    knn_matches = match_features(des1, des2)
    good_matches, _ = filter_matches_ratio(knn_matches, ratio)
    return good_matches


def get_matched_points(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns matched pixel locations as two Nx2 arrays.

    pts1[i] is the point in image 1.
    pts2[i] is the matching point in image 2.
    """
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

    return pts1, pts2


def extract_and_match_features(
    im1: np.ndarray,
    im2: np.ndarray,
    ratio: float = RATIO_THRESHOLD,
) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
    """
    Extracts features from two images and returns ratio-test-filtered matches.
    """
    kp1, desc1 = extract_features(im1)
    kp2, desc2 = extract_features(im2)

    good_matches = match_and_filter(desc1, desc2, ratio)

    return kp1, kp2, good_matches
