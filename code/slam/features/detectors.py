"""Feature detector creation and descriptor extraction utilities."""

from typing import Literal, cast

import cv2
import numpy as np

FeatureType = Literal["sift", "orb", "akaze"]

DEFAULT_FEATURE_TYPE: FeatureType = "sift"
DEFAULT_NUM_FEATURES = 1000
DEFAULT_ORB_NUM_FEATURES = 3000

_cached_detectors: dict[tuple[FeatureType, int | None], cv2.Feature2D] = {}


def normalize_feature_type(feature_type: str) -> FeatureType:
    """Normalize and validate a supported feature-detector name.

    Args:
        feature_type: Requested detector name.

    Returns:
        Lowercase validated detector name.

    Raises:
        ValueError: If the detector type is unsupported.
    """
    normalized = feature_type.lower()

    if normalized not in {"sift", "orb", "akaze"}:
        raise ValueError(f"Unsupported feature type: {feature_type!r}.")

    return cast(FeatureType, normalized)


def create_detector(
    feature_type: FeatureType = DEFAULT_FEATURE_TYPE,
    num_features: int | None = None,
) -> cv2.Feature2D:
    """Create an OpenCV feature detector and descriptor extractor.

    SIFT produces floating-point descriptors for L2 matching. ORB and AKAZE
    produce binary descriptors for Hamming-distance matching. ``num_features``
    controls SIFT and ORB and is ignored for AKAZE.

    Args:
        feature_type: Detector implementation to create.
        num_features: Requested maximum number of SIFT or ORB features.

    Returns:
        Configured OpenCV feature detector.

    Raises:
        ValueError: If ``num_features`` is not positive when provided.
    """
    normalized_type = normalize_feature_type(feature_type)

    if num_features is not None and num_features <= 0:
        raise ValueError("num_features must be positive when provided.")

    if normalized_type == "sift":
        nfeatures = DEFAULT_NUM_FEATURES if num_features is None else num_features
        return cv2.SIFT_create(nfeatures=nfeatures)

    if normalized_type == "orb":
        nfeatures = DEFAULT_ORB_NUM_FEATURES if num_features is None else num_features
        return cv2.ORB_create(nfeatures=nfeatures)

    return cv2.AKAZE_create()


def extract_features(
    image: np.ndarray,
    feature_type: FeatureType = DEFAULT_FEATURE_TYPE,
    num_features: int | None = None,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect keypoints and compute descriptors for one image.

    Args:
        image: Input image, usually a grayscale ``uint8`` image.
        feature_type: Detector and descriptor implementation to use.
        num_features: Requested maximum number of SIFT or ORB features.
            Ignored for AKAZE.

    Returns:
        Detected keypoints and their descriptor matrix.

    Raises:
        RuntimeError: If no descriptors are detected in the image.
    """
    normalized_type = normalize_feature_type(feature_type)

    effective_num_features = None if normalized_type == "akaze" else num_features
    cache_key = (normalized_type, effective_num_features)

    if cache_key not in _cached_detectors:
        _cached_detectors[cache_key] = create_detector(
            feature_type=normalized_type,
            num_features=effective_num_features,
        )

    detector = _cached_detectors[cache_key]
    keypoints, descriptors = detector.detectAndCompute(image, None)

    if descriptors is None:
        raise RuntimeError(f"No descriptors found using {normalized_type.upper()}.")

    return keypoints, descriptors
