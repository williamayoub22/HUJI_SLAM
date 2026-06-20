"""Diagnostics for inspecting bundle-adjustment projection factors."""

from dataclasses import dataclass

import gtsam
from gtsam.symbol_shorthand import C, Q

from slam.ba.gtsam_utils import (
    stereo_image_distances,
    stereo_point_from_triplet,
)
from slam.ba.results import (
    BundleAdjustmentResult,
    ProjectionFactorMetadata,
)
from slam.tracking_database import TrackingDB


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
