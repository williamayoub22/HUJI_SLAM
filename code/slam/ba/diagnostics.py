import gtsam
import numpy as np

from slam.ba.results import ProjectionFactorMetadata
from slam.ba.results import ProjectionEvaluation, LargestFactorDiagnostic, BundleAdjustmentResult
from slam.tracking_database import TrackingDB
from slam.ba.gtsam_utils import stereo_point_from_triplet, stereo_image_projection_distances
from gtsam.symbol_shorthand import C, Q

def factor_error(
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    factor_index: int,
) -> float:
    return graph.at(factor_index).error(values)


def find_largest_initial_projection_factor(
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    projection_factors: list[ProjectionFactorMetadata],
) -> ProjectionFactorMetadata:
    """
    Returns the projection factor with largest initial error.
    """
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
    K: gtsam.Cal3_S2Stereo,
) -> ProjectionEvaluation:
    """
    Evaluate one existing stereo factor using a particular pose/landmark state.
    """
    camera = gtsam.StereoCamera(pose, K)
    projection = camera.project(landmark)

    left_distance, right_distance = stereo_image_projection_distances(
        measurement,
        projection,
    )

    return ProjectionEvaluation(
        factor_error=float(factor.error(values)),
        projection=projection,
        left_distance_pixels=left_distance,
        right_distance_pixels=right_distance,
    )


def analyze_largest_initial_projection_factor(
    db: TrackingDB,
    K: gtsam.Cal3_S2Stereo,
    result: BundleAdjustmentResult,
) -> LargestFactorDiagnostic:
    """
    Find the stereo projection factor with the largest initial error, then
    compare its projection before and after bundle adjustment.
    """
    if not result.projection_factor_metadata:
        raise RuntimeError("No stereo projection factors found.")

    initial_errors = []

    for metadata in result.projection_factor_metadata:
        factor = result.graph.at(metadata.factor_index)
        initial_errors.append(float(factor.error(result.initial)))

    largest_metadata_index = int(np.argmax(initial_errors))
    metadata = result.projection_factor_metadata[largest_metadata_index]

    factor = result.graph.at(metadata.factor_index)

    frame_id = metadata.frame_id
    track_id = metadata.track_id

    measurement = stereo_point_from_triplet(
        db.link_triplet(frame_id, track_id)
    )

    initial_pose = result.initial.atPose3(C(frame_id))
    initial_landmark = result.initial.atPoint3(Q(track_id))

    optimized_pose = result.optimized.atPose3(C(frame_id))
    optimized_landmark = result.optimized.atPoint3(Q(track_id))

    initial_eval = evaluate_projection_factor(
        factor=factor,
        pose=initial_pose,
        landmark=initial_landmark,
        measurement=measurement,
        values=result.initial,
        K=K,
    )

    optimized_eval = evaluate_projection_factor(
        factor=factor,
        pose=optimized_pose,
        landmark=optimized_landmark,
        measurement=measurement,
        values=result.optimized,
        K=K,
    )

    return LargestFactorDiagnostic(
        factor_index=metadata.factor_index,
        frame_id=frame_id,
        track_id=track_id,
        measurement=measurement,
        initial=initial_eval,
        optimized=optimized_eval,
    )
