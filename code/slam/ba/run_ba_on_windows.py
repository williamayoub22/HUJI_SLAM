from dataclasses import dataclass
from collections import defaultdict

import gtsam
import numpy as np
from gtsam.symbol_shorthand import C, Q

from slam.tracking_database import TrackingDB
from .optimization import BundleAdjustmentResult, optimize_bundle_window
from .results import BundleWindowSolution
from .window_selection import bundle_windows_from_keyframes


def solve_all_bundle_windows(
    db: TrackingDB,
    global_camera_matrices: np.ndarray,
    K: gtsam.Cal3_S2Stereo,
    keyframes: list[int],
    min_track_observations: int = 2,
    measurement_sigma_pixels: float = 1.0,
) -> list[BundleWindowSolution]:
    """
    Solve one independent local BA problem for every adjacent keyframe pair.
    """
    solutions = []

    for window_frames in bundle_windows_from_keyframes(keyframes):
        result = optimize_bundle_window(
            db=db,
            global_camera_matrices=global_camera_matrices,
            K=K,
            window_frames=window_frames,
            min_track_observations=min_track_observations,
            measurement_sigma_pixels=measurement_sigma_pixels,
        )

        solutions.append(
            BundleWindowSolution(
                start_frame=window_frames[0],
                end_frame=window_frames[-1],
                result=result,
            )
        )

    return solutions