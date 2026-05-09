
import cv2
import numpy as np

from typing import Sequence, List, Tuple

KNN_NEIGHBORS = 2

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
