"""Load stereo camera calibration from KITTI-style calibration files."""

from pathlib import Path

import numpy as np

from src.slam.config import CALIBRATION_PATH

DEFAULT_CALIBRATION_PATH = CALIBRATION_PATH


def _parse_projection_matrix(
    line: str,
    label: str,
) -> np.ndarray:
    """Parse one labeled 3x4 projection matrix from a calibration-file line.

    Args:
        line: Calibration-file line beginning with ``label``.
        label: Expected matrix label, such as ``"P0:"``.

    Returns:
        Projection matrix with shape ``(3, 4)``.

    Raises:
        ValueError: If the line does not contain exactly 12 matrix values.
    """
    values = line.removeprefix(label).split()

    if len(values) != 12:
        raise ValueError(f"Expected 12 values for calibration entry {label}, got {len(values)}.")

    return np.asarray(values, dtype=float).reshape(3, 4)


def read_stereo_calibration(
    filepath: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load left and right rectified projection matrices from a KITTI file.

    Args:
        filepath: Path to a KITTI-style ``calib.txt`` file. Uses the configured
            sequence calibration file when omitted.

    Returns:
        Left and right camera projection matrices, each with shape ``(3, 4)``.

    Raises:
        FileNotFoundError: If the calibration file does not exist.
        ValueError: If required stereo calibration entries are missing or invalid.
    """
    calibration_path = DEFAULT_CALIBRATION_PATH if filepath is None else Path(filepath)

    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    entries: dict[str, np.ndarray] = {}

    for line in calibration_path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()

        if not stripped_line:
            continue

        for label in ("P0:", "P1:"):
            if stripped_line.startswith(label):
                entries[label] = _parse_projection_matrix(
                    stripped_line,
                    label,
                )

    if "P0:" not in entries or "P1:" not in entries:
        raise ValueError(f"Expected P0: and P1: entries in calibration file {calibration_path}.")

    return entries["P0:"], entries["P1:"]
