"""Load stereo images and stereo feature matches."""

from pathlib import Path

import cv2
import numpy as np

from slam.config import LEFT_IMAGES_DIR, RIGHT_IMAGES_DIR
from slam.features.detectors import FeatureType
from slam.features.matching import extract_and_match_features

DEFAULT_FRAME_INDEX = 0
IMAGE_FILENAME_FORMAT = "{frame_id:06d}.png"


def read_images(
    frame_id: int = DEFAULT_FRAME_INDEX,
    left_images_dir: Path = LEFT_IMAGES_DIR,
    right_images_dir: Path = RIGHT_IMAGES_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one rectified stereo image pair in grayscale.

    Args:
        frame_id: Zero-based frame index.
        left_images_dir: Directory containing left-camera images.
        right_images_dir: Directory containing right-camera images.

    Returns:
        Left and right grayscale images.

    Raises:
        ValueError: If ``frame_id`` is negative.
        FileNotFoundError: If either stereo image cannot be loaded.
    """
    if frame_id < 0:
        raise ValueError(f"frame_id must be non-negative, got {frame_id}.")

    filename = IMAGE_FILENAME_FORMAT.format(frame_id=frame_id)
    left_path = left_images_dir / filename
    right_path = right_images_dir / filename

    left_image = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

    if left_image is None or right_image is None:
        raise FileNotFoundError(
            f"Could not load stereo images:\n  left:  {left_path}\n  right: {right_path}"
        )

    return left_image, right_image


def load_matches_between_images(
    frame_id: int,
    feature_type: FeatureType = "sift",
    num_features: int = 1000,
    use_ratio_test: bool = False,
) -> tuple[
    list[cv2.KeyPoint],
    list[cv2.KeyPoint],
    np.ndarray,
    np.ndarray,
    list[cv2.DMatch],
]:
    """Load one stereo pair, extract features, and match their descriptors.

    Args:
        frame_id: Zero-based stereo-frame index.
        feature_type: Feature detector and descriptor type.
        num_features: Requested number of detected features.
        use_ratio_test: Whether to apply Lowe's ratio test.

    Returns:
        Left keypoints, right keypoints, left image, right image, and accepted
        stereo matches.
    """
    left_image, right_image = read_images(frame_id)

    left_keypoints, right_keypoints, matches = extract_and_match_features(
        left_image,
        right_image,
        feature_type=feature_type,
        num_features=num_features,
        use_ratio_test=use_ratio_test,
    )

    return (
        left_keypoints,
        right_keypoints,
        left_image,
        right_image,
        matches,
    )
