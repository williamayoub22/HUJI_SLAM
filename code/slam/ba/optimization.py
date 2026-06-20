import gtsam
import numpy as np

from slam.ba.graph_builder import build_local_bundle_graph
from slam.ba.results import BundleAdjustmentResult
from slam.tracking_database import TrackingDB

# Helper to validate that the initial and graph keys match
def _validate_graph_keys(
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    window_frames: list[int],
) -> None:
    """Fail early when `initial` contains variables unused by the graph."""
    factor_keys: set[int] = set()
    for factor_index in range(graph.size()):
        factor_keys.update(int(key) for key in graph.at(factor_index).keys())

    initial_keys = {int(key) for key in initial.keys()}
    unused_initial_keys = initial_keys - factor_keys
    missing_initial_keys = factor_keys - initial_keys

    if missing_initial_keys:
        raise RuntimeError(
            "The factor graph references variables absent from the initial "
            f"estimate for window {window_frames[0]} -> {window_frames[-1]}: "
            f"{sorted(missing_initial_keys)}"
        )

    if unused_initial_keys:
        raise RuntimeError(
            "The initial estimate contains variables that are not referenced "
            "by any factor, which makes the GTSAM ordering inconsistent for "
            f"window {window_frames[0]} -> {window_frames[-1]}. "
            f"Unused keys: {sorted(unused_initial_keys)}"
        )



def optimize_bundle_window(
    db: TrackingDB,
    global_camera_matrices: np.ndarray,
    K: gtsam.Cal3_S2Stereo,
    window_frames: list[int],
    min_track_observations: int = 2,
    measurement_sigma_pixels: float = 1.0,
) -> BundleAdjustmentResult:
    """
    Builds and optimizes one local bundle adjustment window.
    """
    graph, initial, track_ids, projection_factor_metadata = build_local_bundle_graph(
        db=db,
        global_camera_matrices=global_camera_matrices,
        K=K,
        window_frames=window_frames,
        min_track_observations=min_track_observations,
        measurement_sigma_pixels=measurement_sigma_pixels,
    )

    initial_error = graph.error(initial)

    _validate_graph_keys(
        graph=graph,
        initial=initial,
        window_frames=window_frames,
    )

    try:
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        optimized = optimizer.optimize()
    except RuntimeError as error:
        raise RuntimeError(
            f"Levenberg-Marquardt failed for bundle window "
            f"{window_frames[0]} -> {window_frames[-1]} "
            f"with {graph.size()} factors and {initial.size()} initial values."
        ) from error

    final_error = graph.error(optimized)
    num_factors = graph.size()

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
