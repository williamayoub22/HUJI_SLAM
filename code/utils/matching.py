
import cv2
import numpy as np

from typing import Sequence

KNN_NEIGHBORS = 2

def match_features(des1: np.ndarray, des2: np.ndarray, k: int = KNN_NEIGHBORS) -> Sequence[Sequence[cv2.DMatch]]:
    """
    Returns all matches without the significance / ratio test.
    For every descriptor in image 1, we keep its nearest neighbor in image 2.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=k)
    return knn_matches
