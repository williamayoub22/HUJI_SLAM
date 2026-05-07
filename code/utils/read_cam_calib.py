import numpy as np
from typing import Tuple, Union
from pathlib import Path

# Construct absolute path to the dataset based on this file's location
# __file__ is code/utils/read_cam_calib.py
# SCRIPT_DIR is code/utils
# SCRIPT_DIR.parent.parent is the root directory
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent / "dataset" / "sequences" / "00"
DEFAULT_FILEPATH = BASE_DIR / "calib.txt"


def read_calib(filepath: Union[str, Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the 3x4 camera projection matrices P1 (left) and P2 (right) from calib.txt.
    If no filepath is provided, it defaults to dataset/sequences/00/calib.txt.
    """
    # Use default if none provided, otherwise ensure it's a Path object
    if filepath is None:
        filepath = DEFAULT_FILEPATH
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Calibration file not found at: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    P1 = None
    P2 = None

    # Try to parse standard KITTI format (P0: ..., P1: ...)
    for line in lines:
        if line.startswith('P0:'):
            P1 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
        elif line.startswith('P1:'):
            P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)

    # Fallback: if prefixes aren't there, just read the first two lines
    if P1 is None or P2 is None:
        P1 = np.array([float(x) for x in lines[0].strip().split()]).reshape(3, 4)
        P2 = np.array([float(x) for x in lines[1].strip().split()]).reshape(3, 4)

    return P1, P2