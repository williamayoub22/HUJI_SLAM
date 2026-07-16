"""Solve independent local bundle-adjustment problems over keyframe windows."""

from time import perf_counter

import gtsam
import numpy as np
from slam.database.facade import SlamDatabase

from .optimization import optimize_bundle_window
from .results import BundleWindowSolution
from .window_selection import bundle_windows_from_keyframes


def solve_all_bundle_windows(
    slam_db: SlamDatabase,
    calibration: gtsam.Cal3_S2Stereo,
    keyframes: list[int],
    min_track_observations: int = 2,
    measurement_sigma_pixels: float = 1.0,
    verbose: bool = False,
) -> list[BundleWindowSolution]:
    """Solve one independent local BA problem for each adjacent keyframe pair.

    Each window is optimized in the coordinate system of its first frame.

    Args:
        slam_db: Unified tracking database containing 2D tracks and poses.
        calibration: Stereo camera calibration.
        keyframes: Ordered keyframe IDs defining local BA windows.
        min_track_observations: Minimum observations required to retain a track.
        measurement_sigma_pixels: Stereo measurement noise standard deviation.
        verbose: Whether to print progress for each solved window.

    Returns:
        One solved bundle-adjustment result for every adjacent keyframe pair.
    """
    windows = bundle_windows_from_keyframes(keyframes)

    if not windows:
        raise ValueError("At least two keyframes are required.")

    solutions: list[BundleWindowSolution] = []
    total_windows = len(windows)

    for window_index, window_frames in enumerate(windows, start=1):
        start_frame = window_frames[0]
        end_frame = window_frames[-1]

        if verbose:
            print(
                f"Window {window_index}/{total_windows}: "
                f"{start_frame} -> {end_frame} "
                f"({len(window_frames)} frames)...",
                flush=True,
            )

        start_time = perf_counter()

        result = optimize_bundle_window(
            slam_db=slam_db,
            calibration=calibration,
            window_frames=window_frames,
            min_track_observations=min_track_observations,
            measurement_sigma_pixels=measurement_sigma_pixels,
        )

        elapsed_seconds = perf_counter() - start_time

        solutions.append(
            BundleWindowSolution(
                start_frame=start_frame,
                end_frame=end_frame,
                result=result,
            )
        )

        if verbose:
            print(
                f"      done in {elapsed_seconds:.1f}s | "
                f"poses={len(window_frames)} | "
                f"landmarks={len(result.track_ids)} | "
                f"factors={result.num_factors} | "
                f"error={result.initial_error:.1f} -> {result.final_error:.1f}",
                flush=True,
            )

    return solutions
