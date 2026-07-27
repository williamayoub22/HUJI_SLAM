"""Load stereo images and stereo feature matches."""

from pathlib import Path

import cv2
import numpy as np
from src.slam.config import LEFT_IMAGES_DIR, RIGHT_IMAGES_DIR
from src.slam.config import LEFT_IMAGES_DIR, RIGHT_IMAGES_DIR

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

