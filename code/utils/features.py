import cv2
import numpy as np

NUM_FEATURES = 1000

def extract_features(img: np.ndarray, num_features: int = NUM_FEATURES):
    """
    Detects keypoints and extracts descriptors using SIFT.
    Returns a tuple of (keypoints, descriptors).
    """
    sift = cv2.SIFT_create(nfeatures=num_features)
    keypoints, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        raise RuntimeError("No descriptors found in image.")

    return keypoints, descriptors

