import cv2
import numpy as np
from typing import Literal, Tuple, List


FeatureType = Literal["sift", "orb"]

DEFAULT_FEATURE_TYPE: FeatureType = "sift"
DEFAULT_NUM_FEATURES = 1000
DEFAULT_ORB_NUM_FEATURES = 3000


_cached_detectors = {}


def create_detector(feature_type: FeatureType = DEFAULT_FEATURE_TYPE, num_features: int | None = None):
    """
    Create a feature detector.

    SIFT:
        - Floating-point descriptors
        - Should be matched with NORM_L2

    ORB:
        - Binary descriptors
        - Should be matched with NORM_HAMMING
    """
    feature_type = feature_type.lower()

    if feature_type == "sift":
        nfeatures = DEFAULT_NUM_FEATURES if num_features is None else num_features
        return cv2.SIFT_create(nfeatures=nfeatures)

    if feature_type == "orb":
        nfeatures = DEFAULT_ORB_NUM_FEATURES if num_features is None else num_features
        return cv2.ORB_create(nfeatures=nfeatures)

    raise ValueError(f"Unsupported feature_type: {feature_type}")


def extract_features(
    img: np.ndarray,
    feature_type: FeatureType = DEFAULT_FEATURE_TYPE,
    num_features: int | None = None,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detect keypoints and extract descriptors using SIFT or ORB.

    Default is SIFT for backward compatibility with ex1/ex2.
    Use feature_type="orb" explicitly in ex3 if needed.
    """
    cache_key = (feature_type, num_features)

    if cache_key not in _cached_detectors:
        _cached_detectors[cache_key] = create_detector(feature_type, num_features)

    detector = _cached_detectors[cache_key]
    keypoints, descriptors = detector.detectAndCompute(img, None)

    if descriptors is None:
        raise RuntimeError(f"No descriptors found using {feature_type.upper()}.")

    return keypoints, descriptors