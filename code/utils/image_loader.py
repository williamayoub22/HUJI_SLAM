
import cv2
import os
import numpy as np

from typing import Tuple
from pathlib import Path

FRAME_INDEX = 0

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "dataset" / "sequences" / "00"
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
