"""Optimize and validate local stereo bundle-adjustment windows."""

import gtsam
import numpy as np

from .. import config
from src.slam.ba.graph_builder import build_local_bundle_graph
from src.slam.ba.results import BundleAdjustmentResult
from src.slam.database.facade import SlamDatabase


def _validate_graph_keys(
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    window_frames: list[int],
) -> None:
    """Validate that graph variables and initial values use identical keys.

    Args:
        graph: Factor graph whose variables are checked.
        initial: Initial variable values supplied to the optimizer.
        window_frames: Frame IDs in the local bundle-adjustment window.

    Raises:
        RuntimeError: If a factor references a missing value or an initial value
            is not referenced by any factor.
    """
    factor_keys: set[int] = set()

    for factor_index in range(graph.size()):
        factor = graph.at(factor_index)
        factor_keys.update(int(key) for key in factor.keys())

    initial_keys = {int(key) for key in initial.keys()}

    unused_initial_keys = initial_keys - factor_keys
    missing_initial_keys = factor_keys - initial_keys

    window_description = f"{window_frames[0]} -> {window_frames[-1]}"

    if missing_initial_keys:
        raise RuntimeError(
            "The factor graph references variables absent from the initial "
            f"estimate for window {window_description}: "
            f"{sorted(missing_initial_keys)}"
        )

    if unused_initial_keys:
        raise RuntimeError(
            "The initial estimate contains variables not referenced by any "
            f"factor for window {window_description}: "
            f"{sorted(unused_initial_keys)}"
        )


def optimize_bundle_window(
    slam_db: SlamDatabase,
    calibration: gtsam.Cal3_S2Stereo,
    window_frames: list[int],
    min_track_observations: int = 2,
    measurement_sigma_pixels: float = config.MEASUREMENT_SIGMA_PIXELS,
) -> BundleAdjustmentResult:
    """Build and optimize one local stereo bundle-adjustment window.

    Args:
        slam_db: Unified tracking database.
        calibration: Stereo camera calibration.
        window_frames: Consecutive frame IDs in the local optimization window.
        min_track_observations: Minimum observations required to retain a track.
        measurement_sigma_pixels: Stereo measurement noise standard deviation.

    Returns:
        Graph, initial and optimized values, factor metadata, and error metrics.

    Raises:
        RuntimeError: If graph validation or Levenberg-Marquardt optimization
            fails.
        ValueError: If the requested window or optimization parameters are invalid.
    """
    graph, initial, track_ids, projection_factor_metadata = build_local_bundle_graph(
        slam_db=slam_db,
        calibration=calibration,
        window_frames=window_frames,
        min_track_observations=min_track_observations,
        measurement_sigma_pixels=measurement_sigma_pixels,
    )

    _validate_graph_keys(
        graph=graph,
        initial=initial,
        window_frames=window_frames,
    )

    initial_error = float(graph.error(initial))
    num_factors = graph.size()

    try:
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        optimized = optimizer.optimize()
    except RuntimeError as error:
        raise RuntimeError(
            "Levenberg-Marquardt failed for bundle window "
            f"{window_frames[0]} -> {window_frames[-1]} "
            f"with {num_factors} factors and {initial.size()} initial values."
        ) from error

    final_error = float(graph.error(optimized))

    return BundleAdjustmentResult(
        window_frames=window_frames,
        graph=graph,
        initial=initial,
        optimized=optimized,
        track_ids=track_ids,
        projection_factor_metadata=projection_factor_metadata,
        initial_error=initial_error,
        final_error=final_error,
        num_factors=num_factors,
        average_initial_error=initial_error / num_factors,
        average_final_error=final_error / num_factors,
    )
