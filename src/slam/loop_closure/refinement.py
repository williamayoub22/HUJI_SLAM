from __future__ import annotations

from dataclasses import dataclass

import cv2
import gtsam
import numpy as np
from gtsam.symbol_shorthand import C

from src.slam.geometry.stereo import (
    make_gtsam_stereo_calibration,
    pose3_from_world_to_camera_extrinsic,
)
from src.slam.pose_graph.constraints import symmetrize
from src.slam.pose_graph.covariance_routing import (
    CovarianceGraph,
    mahalanobis_squared,
)
from .. import config
from src.slam.io.calibration import read_stereo_calibration
from src.slam.loop_closure.consensus import ConsensusMatchResult


SOURCE_POSE_PRIOR_SIGMA = 1e-9

@dataclass(frozen=True)
class RelativePoseEstimate:
    """Relative pose and covariance refined by two-frame stereo BA."""

    consensus_result: ConsensusMatchResult
    relative_pose: gtsam.Pose3
    covariance: np.ndarray
    initial_error: float
    final_error: float
    num_landmarks: int


def _stereo_measurement(
    left_point: np.ndarray,
    right_point: np.ndarray,
) -> gtsam.StereoPoint2:
    """Convert rectified left/right image coordinates to GTSAM's stereo format:
    (u_left, u_right, v).
    """
    return gtsam.StereoPoint2(
        float(left_point[0]),
        float(right_point[0]),
        float(left_point[1]),
    )


def refine_relative_pose_with_bundle_adjustment(
    consensus_result: ConsensusMatchResult,
) -> RelativePoseEstimate:
    """Refine one verified loop closure with a two-frame stereo bundle adjustment.

    The source camera is fixed as the local world frame. Therefore, the
    optimized target pose directly represents the source-to-target relative
    pose in camera-to-world Pose3 convention.
    """
    if not consensus_result.success:
        raise ValueError("Bundle refinement requires a successful consensus match.")

    if consensus_result.source_to_target_transform is None:
        raise ValueError("Missing PnP/RANSAC transform.")

    if consensus_result.inlier_mask is None:
        raise ValueError("Missing consensus inlier mask.")

    if any(
        array is None
        for array in (
            consensus_result.points_3d,
            consensus_result.source_left,
            consensus_result.source_right,
            consensus_result.target_left,
            consensus_result.target_right,
        )
    ):
        raise ValueError("Consensus result does not retain the four-view correspondences.")

    inlier_mask = np.asarray(
        consensus_result.inlier_mask,
        dtype=bool,
    ).reshape(-1)

    points_3d = np.asarray(consensus_result.points_3d, dtype=float)[inlier_mask]
    source_left = np.asarray(consensus_result.source_left, dtype=float)[inlier_mask]
    source_right = np.asarray(consensus_result.source_right, dtype=float)[inlier_mask]
    target_left = np.asarray(consensus_result.target_left, dtype=float)[inlier_mask]
    target_right = np.asarray(consensus_result.target_right, dtype=float)[inlier_mask]

    if len(points_3d) < 4:
        raise RuntimeError("Need at least four inliers for two-frame BA.")

    projection_left, projection_right = read_stereo_calibration()
    stereo_calibration = make_gtsam_stereo_calibration(
        projection_left,
        projection_right,
    )

    source_key = C(consensus_result.candidate.source_frame)
    target_key = C(consensus_result.candidate.target_frame)

    # The source camera defines the local world frame.
    source_pose = gtsam.Pose3()

    # PnP gives a source-camera -> target-camera world-to-camera transform.
    # Reuse the project utility that converts it to GTSAM's camera-to-world Pose3.
    target_pose_initial = pose3_from_world_to_camera_extrinsic(
        consensus_result.source_to_target_transform
    )

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    initial_values.insert(source_key, source_pose)
    initial_values.insert(target_key, target_pose_initial)

    source_prior_noise = gtsam.noiseModel.Isotropic.Sigma(
        6,
        SOURCE_POSE_PRIOR_SIGMA,
    )
    graph.add(
        gtsam.PriorFactorPose3(
            source_key,
            source_pose,
            source_prior_noise,
        )
    )

    stereo_noise = gtsam.noiseModel.Isotropic.Sigma(
        3,
        config.MEASUREMENT_SIGMA_PIXELS,
    )

    for landmark_index, point_3d in enumerate(points_3d):
        landmark_key = gtsam.symbol("l", landmark_index)

        initial_values.insert(
            landmark_key,
            gtsam.Point3(
                float(point_3d[0]),
                float(point_3d[1]),
                float(point_3d[2]),
            ),
        )

        source_measurement = _stereo_measurement(
            source_left[landmark_index],
            source_right[landmark_index],
        )
        target_measurement = _stereo_measurement(
            target_left[landmark_index],
            target_right[landmark_index],
        )

        graph.add(
            gtsam.GenericStereoFactor3D(
                source_measurement,
                stereo_noise,
                source_key,
                landmark_key,
                stereo_calibration,
            )
        )
        graph.add(
            gtsam.GenericStereoFactor3D(
                target_measurement,
                stereo_noise,
                target_key,
                landmark_key,
                stereo_calibration,
            )
        )

    initial_error = graph.error(initial_values)

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_values,
    )
    optimized_values = optimizer.optimize()

    final_error = graph.error(optimized_values)

    optimized_source_pose = optimized_values.atPose3(source_key)
    optimized_target_pose = optimized_values.atPose3(target_key)

    relative_pose = optimized_source_pose.between(optimized_target_pose)

    marginals = gtsam.Marginals(
        graph,
        optimized_values,
    )

    # The source pose is effectively fixed, so target covariance is the
    # relative-pose covariance. This is the correct covariance for the
    # later BetweenFactorPose3 loop constraint.
    covariance = symmetrize(marginals.marginalCovariance(target_key))

    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues) <= 0:
        raise RuntimeError(
            "Two-frame BA produced a non-positive relative-pose covariance: "
            f"minimum eigenvalue = {np.min(eigenvalues):.3e}"
        )

    return RelativePoseEstimate(
        consensus_result=consensus_result,
        relative_pose=relative_pose,
        covariance=covariance,
        initial_error=initial_error,
        final_error=final_error,
        num_landmarks=len(points_3d),
    )


