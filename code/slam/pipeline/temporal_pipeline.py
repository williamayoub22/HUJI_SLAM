from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from ..io.image_loader import read_images
from ..features.detectors import FeatureType
from ..features.matching import (
    RATIO_THRESHOLD,
    extract_and_match_features,
    get_matched_points,
)


@dataclass
class TemporalMatchData:
    """Container for temporal matches between two left-camera frames."""
    left0_img: np.ndarray
    left1_img: np.ndarray
    kp_left0: List[cv2.KeyPoint]
    kp_left1: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]
    left0_pts: np.ndarray
    left1_pts: np.ndarray


def match_left_frames(
    frame_idx0: int,
    frame_idx1: int,
    feature_type: FeatureType,
    num_features: int,
    use_ratio_test: bool = True,
    ratio_threshold: float = RATIO_THRESHOLD,
) -> TemporalMatchData:
    """
    Matches features between the left images of two frames.

    This is temporal matching, not stereo matching, so no vertical-deviation
    filtering is applied here.
    """
    left0_img, _ = read_images(frame_idx0)
    left1_img, _ = read_images(frame_idx1)

    kp_left0, kp_left1, matches = extract_and_match_features(
        left0_img,
        left1_img,
        feature_type=feature_type,
        num_features=num_features,
        ratio_threshold=ratio_threshold,
        use_ratio_test=use_ratio_test,
    )

    left0_pts, left1_pts = get_matched_points(kp_left0, kp_left1, matches)

    return TemporalMatchData(
        left0_img=left0_img,
        left1_img=left1_img,
        kp_left0=kp_left0,
        kp_left1=kp_left1,
        matches=matches,
        left0_pts=left0_pts,
        left1_pts=left1_pts,
    )
