
import cv2
import os
import numpy as np

from typing import Tuple
from pathlib import Path

from ..features.matching import extract_and_match_features

FRAME_INDEX = 0

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT / "dataset" / "sequences" / "00"

LEFT_IMG_DIR = 'image_0'
RIGHT_IMG_DIR = 'image_1'
IMG_FILENAME_FORMAT = '{:06d}.png'


def read_images(idx: int = FRAME_INDEX) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the stereo pair for a given index using pinned path formats."""
    img_name = IMG_FILENAME_FORMAT.format(idx)

    left_path = os.path.join(BASE_DIR, LEFT_IMG_DIR, img_name)
    right_path = os.path.join(BASE_DIR, RIGHT_IMG_DIR, img_name)

    img1 = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not read images. Check paths:\n{left_path}\n{right_path}")

    return img1, img2


def load_matches_between_images(
    frame_idx: int,
    feature_type: str = "sift",
    num_features: int = 1000,
    use_ratio_test: bool = False,
):
    """
    Loads a stereo image pair, extracts features, and computes feature matches.

    Returns:
        kp_left, kp_right: Detected keypoints.
        left_img, right_img: Stereo images.
        matches: Feature matches between the images.
    """
    left_img, right_img = read_images(frame_idx)

    kp_left, kp_right, matches = extract_and_match_features(
        left_img,
        right_img,
        feature_type=feature_type,
        num_features=num_features,
        use_ratio_test=use_ratio_test,
    )

    return kp_left, kp_right, left_img, matches, right_img

