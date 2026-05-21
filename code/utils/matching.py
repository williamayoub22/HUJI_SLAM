
import cv2
import numpy as np

from typing import Sequence, List, Tuple, Any

from utils.features import extract_features
from cv2 import DMatch

KNN_NEIGHBORS = 2
RATIO_THRESHOLD = 0.7

def match_features(des1: np.ndarray, des2: np.ndarray, k: int = KNN_NEIGHBORS) -> Sequence[Sequence[cv2.DMatch]]:
    """
    Returns all matches without the significance / ratio test.
    For every descriptor in image 1, we keep its nearest neighbor in image 2.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=k)
    return knn_matches

def get_matched_points(
        kp_left: List[cv2.KeyPoint],
        kp_right: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns matched pixel locations as two Nx2 arrays:
    left_pts[i] = (x_left, y_left)
    right_pts[i] = (x_right, y_right)
    """
    left_pts = np.array([kp_left[m.queryIdx].pt for m in matches])
    right_pts = np.array([kp_right[m.trainIdx].pt for m in matches])

    return left_pts, right_pts

def extract_and_match_features(
    im1: np.ndarray,
    im2: np.ndarray,
) -> tuple[Any, Any, Sequence[Sequence[DMatch]]]:
    """Extracts local features from two images and matches their descriptors."""
    kp1, desc1 = extract_features(im1)
    kp2, desc2 = extract_features(im2)
    matches = match_features(desc1, desc2)
    return kp1, kp2, matches

def filter_matches_ratio(knn_matches: list, ratio: float = RATIO_THRESHOLD):
    """LOGIC: Filters matches using Lowe's Ratio Test."""
    good_matches = []
    rejected_matches = []

    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
        else:
            rejected_matches.append(m)

    return good_matches, rejected_matches

