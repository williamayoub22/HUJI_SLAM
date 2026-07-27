from dataclasses import dataclass
import gtsam
import numpy as np
from src.slam.io.calibration import read_stereo_calibration
from src.slam.geometry.stereo import make_gtsam_stereo_calibration
from src.slam.config import GLOBAL_CAMERA_MATRICES_PATH
from src.slam.bundle_adjustment.window_selection import choose_keyframes_by_motion
from src.slam.bundle_adjustment.window_solver import solve_all_bundle_windows
from src.slam.bundle_adjustment.types import BundleWindowSolution

@dataclass
class BundleAnalysisContext:
    calibration: gtsam.Cal3_S2Stereo
    keyframes: list[int]
    solutions: list[BundleWindowSolution]

"""Diagnostics for inspecting bundle-adjustment projection factors."""

from dataclasses import dataclass

import gtsam
from gtsam.symbol_shorthand import C, Q
from src.slam.geometry.stereo import (
    stereo_image_distances,
    stereo_point_from_triplet,
)
from src.slam.bundle_adjustment.types import (
    BundleAdjustmentResult,
    ProjectionFactorMetadata,
)
from src.slam.data.tracking_db import TrackingDB


@dataclass(frozen=True)
class ProjectionEvaluation:
    """Store one stereo factor evaluation at a particular state.

    Attributes:
        factor_error: GTSAM factor error at the supplied values.
        projection: Predicted stereo image measurement.
        left_distance_pixels: Left-image distance from projection to measurement.
        right_distance_pixels: Right-image distance from projection to measurement.
    """

    factor_error: float
    projection: gtsam.StereoPoint2
    left_distance_pixels: float
    right_distance_pixels: float


@dataclass(frozen=True)
class LargestFactorDiagnostic:
    """Compare one projection factor before and after optimization.

    Attributes:
        factor_index: Index of the selected factor in the graph.
        frame_id: Camera frame associated with the factor.
        track_id: Landmark track associated with the factor.
        measurement: Observed stereo image measurement.
        initial: Evaluation using initial values.
        optimized: Evaluation using optimized values.
    """

    factor_index: int
    frame_id: int
    track_id: int
    measurement: gtsam.StereoPoint2
    initial: ProjectionEvaluation
    optimized: ProjectionEvaluation


def factor_error(
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    factor_index: int,
) -> float:
    """Evaluate one graph factor at the supplied variable values.

    Args:
        graph: Factor graph containing the factor.
        values: Variable assignment used for factor evaluation.
        factor_index: Index of the factor in ``graph``.

    Returns:
        GTSAM error of the selected factor.
    """
    return float(graph.at(factor_index).error(values))


def find_largest_initial_projection_factor(
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    projection_factors: list[ProjectionFactorMetadata],
) -> ProjectionFactorMetadata:
    """Return metadata for the projection factor with the largest initial error.

    Args:
        graph: Bundle-adjustment factor graph.
        initial: Initial pose and landmark estimates.
        projection_factors: Metadata for projection factors in ``graph``.

    Returns:
        Metadata of the projection factor with the largest initial error.

    Raises:
        ValueError: If no projection factors are provided.
    """
    if not projection_factors:
        raise ValueError("No projection factors were provided.")

    return max(
        projection_factors,
        key=lambda meta: factor_error(graph, initial, meta.factor_index),
    )


def evaluate_projection_factor(
    factor: gtsam.NonlinearFactor,
    pose: gtsam.Pose3,
    landmark: gtsam.Point3,
    measurement: gtsam.StereoPoint2,
    values: gtsam.Values,
    calibration: gtsam.Cal3_S2Stereo,
) -> ProjectionEvaluation:
    """Evaluate a stereo factor and its left/right reprojection distances.

    Args:
        factor: Stereo factor whose GTSAM error is evaluated.
        pose: Camera pose used to project the landmark.
        landmark: Landmark position in global coordinates.
        measurement: Observed stereo measurement.
        values: Variable assignment used for factor evaluation.
        calibration: Stereo camera calibration.

    Returns:
        Factor error, projected measurement, and left/right pixel distances.
    """
    camera = gtsam.StereoCamera(pose, calibration)
    projection = camera.project(landmark)

    left_distance, right_distance = stereo_image_distances(measurement, projection)

    return ProjectionEvaluation(
        factor_error=float(factor.error(values)),
        projection=projection,
        left_distance_pixels=left_distance,
        right_distance_pixels=right_distance,
    )


def analyze_largest_initial_projection_factor(
    db: TrackingDB,
    calibration: gtsam.Cal3_S2Stereo,
    result: BundleAdjustmentResult,
) -> LargestFactorDiagnostic:
    """Compare the worst initial projection factor before and after optimization.

    The factor is selected using the initial bundle-adjustment estimate, so the
    result helps identify the measurement or geometry causing the largest
    initial reprojection inconsistency.

    Args:
        db: Tracking database containing stereo observations.
        calibration: Stereo camera calibration.
        result: Bundle-adjustment graph, values, and factor metadata.

    Returns:
        Before-and-after diagnostic data for the worst initial projection factor.

    Raises:
        RuntimeError: If the result contains no projection factors.
    """
    if not result.projection_factor_metadata:
        raise RuntimeError("No stereo projection factors found.")

    metadata = find_largest_initial_projection_factor(
        graph=result.graph,
        initial=result.initial,
        projection_factors=result.projection_factor_metadata,
    )

    factor = result.graph.at(metadata.factor_index)
    measurement = stereo_point_from_triplet(db.link_triplet(metadata.frame_id, metadata.track_id))

    initial_evaluation = evaluate_projection_factor(
        factor=factor,
        pose=result.initial.atPose3(C(metadata.frame_id)),
        landmark=result.initial.atPoint3(Q(metadata.track_id)),
        measurement=measurement,
        values=result.initial,
        calibration=calibration,
    )

    optimized_evaluation = evaluate_projection_factor(
        factor=factor,
        pose=result.optimized.atPose3(C(metadata.frame_id)),
        landmark=result.optimized.atPoint3(Q(metadata.track_id)),
        measurement=measurement,
        values=result.optimized,
        calibration=calibration,
    )

    return LargestFactorDiagnostic(
        factor_index=metadata.factor_index,
        frame_id=metadata.frame_id,
        track_id=metadata.track_id,
        measurement=measurement,
        initial=initial_evaluation,
        optimized=optimized_evaluation,
    )



def _get_result_value(result, possible_names):
    """Read a value from either a dictionary or a result object."""
    for name in possible_names:
        if isinstance(result, dict) and name in result:
            return result[name]
        if hasattr(result, name):
            return getattr(result, name)
    raise AttributeError(f'Could not find any of {possible_names} in bundle-window result.')

def compute_bundle_window_factor_errors(bundle_results):
    """Compute mean factor error before and after BA for each window.

    Each result must contain:
        - the window factor graph;
        - the initial Values;
        - the optimized Values.

    The function supports several common field names so it can work with
    either dictionaries or result objects.
    """
    window_ids = []
    initial_mean_errors = []
    optimized_mean_errors = []
    for window_index, result in enumerate(bundle_results):
        try:
            graph = _get_result_value(result, ('graph', 'factor_graph', 'bundle_graph'))
            initial_values = _get_result_value(result, ('initial_values', 'initial', 'initial_estimate'))
            optimized_values = _get_result_value(result, ('optimized_values', 'result_values', 'optimized', 'solution'))
            number_of_factors = int(graph.size())
            if number_of_factors == 0:
                continue
            initial_total_error = float(graph.error(initial_values))
            optimized_total_error = float(graph.error(optimized_values))
            window_ids.append(window_index)
            initial_mean_errors.append(initial_total_error / number_of_factors)
            optimized_mean_errors.append(optimized_total_error / number_of_factors)
        except (AttributeError, KeyError, ValueError, RuntimeError) as error:
            print(f'WARNING: Could not compute errors for bundle window {window_index}: {error}')
    return (np.asarray(window_ids, dtype=int), np.asarray(initial_mean_errors, dtype=float), np.asarray(optimized_mean_errors, dtype=float))

def extract_bundle_window_average_errors(bundle_solutions):
    """Extract normalized factor errors from solved BA windows."""
    window_ids = []
    initial_average_errors = []
    final_average_errors = []
    for window_index, solution in enumerate(bundle_solutions, start=1):
        result = solution.result
        if result.num_factors <= 0:
            continue
        initial_error = float(result.average_initial_error)
        final_error = float(result.average_final_error)
        if not np.isfinite(initial_error) or not np.isfinite(final_error):
            continue
        window_ids.append(window_index)
        initial_average_errors.append(initial_error)
        final_average_errors.append(final_error)
    return (np.asarray(window_ids, dtype=int), np.asarray(initial_average_errors, dtype=float), np.asarray(final_average_errors, dtype=float))

def compute_bundle_window_errors(bundle_solutions):
    window_ids = []
    initial_average_errors = []
    final_average_errors = []
    initial_median_proj_errors = []
    final_median_proj_errors = []
    for window_index, solution in enumerate(bundle_solutions, start=1):
        result = solution.result
        if result.num_factors <= 0:
            continue
        initial_error = float(result.average_initial_error)
        final_error = float(result.average_final_error)
        if not np.isfinite(initial_error) or not np.isfinite(final_error):
            continue
        init_proj_errors = []
        final_proj_errors = []
        for metadata in result.projection_factor_metadata:
            factor = result.graph.at(metadata.factor_index)
            e_init = factor.unwhitenedError(result.initial)
            e_final = factor.unwhitenedError(result.optimized)
            init_proj_errors.append(np.linalg.norm(e_init))
            final_proj_errors.append(np.linalg.norm(e_final))
        window_ids.append(window_index)
        initial_average_errors.append(initial_error)
        final_average_errors.append(final_error)
        initial_median_proj_errors.append(np.median(init_proj_errors) if init_proj_errors else 0.0)
        final_median_proj_errors.append(np.median(final_proj_errors) if final_proj_errors else 0.0)
    return (np.asarray(window_ids, dtype=int), np.asarray(initial_average_errors, dtype=float), np.asarray(final_average_errors, dtype=float), np.asarray(initial_median_proj_errors, dtype=float), np.asarray(final_median_proj_errors, dtype=float))

def prepare_bundle_adjustment_analysis(database) -> BundleAnalysisContext:
    """Select motion-based keyframes and solve all BA windows once."""
    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)
    pnp_extrinsics = np.asarray(np.load(GLOBAL_CAMERA_MATRICES_PATH), dtype=float)
    keyframes = choose_keyframes_by_motion(poses=pnp_extrinsics, min_gap=11, max_gap=20, min_translation=5.0, min_rotation_deg=12.0, target_translation=5.0, target_rotation_deg=12.0)
    print(f'Selected {len(keyframes)} motion-based keyframes.')
    print(f'First keyframes: {keyframes[:10]}')
    print(f'Last keyframes:  {keyframes[-10:]}')
    solutions = solve_all_bundle_windows(slam_db=database, calibration=calibration, keyframes=keyframes, verbose=True)
    return BundleAnalysisContext(calibration=calibration, keyframes=keyframes, solutions=solutions)