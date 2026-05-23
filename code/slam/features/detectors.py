import cv2
import numpy as np

NUM_FEATURES = 3000

# Cached detector — created once, reused across all calls
_cached_detector = None
_cached_nfeatures = None


def extract_features(img: np.ndarray, num_features: int = NUM_FEATURES):
    """
    Detects keypoints and extracts descriptors using ORB.
    The detector is cached so it is only created once for a given num_features.
    Returns a tuple of (keypoints, descriptors).
    """
    global _cached_detector, _cached_nfeatures
    if _cached_detector is None or _cached_nfeatures != num_features:
        _cached_detector = cv2.ORB_create(nfeatures=num_features)
        _cached_nfeatures = num_features

    keypoints, descriptors = _cached_detector.detectAndCompute(img, None)

    if descriptors is None:
        raise RuntimeError("No descriptors found in image.")

    return keypoints, descriptors
