from dataclasses import dataclass
import gtsam
import numpy as np


@dataclass
class TrackReprojectionResult:
    """
    Holds the numerical results of Exercise 5.1.
    """
    track_id: int
    frame_ids: list[int]
    landmark_global: gtsam.Point3
    reprojection_errors: np.ndarray
    factor_errors: np.ndarray
    covariance: np.ndarray


@dataclass
class ProjectionFactorMetadata:
    factor_index: int
    frame_id: int
    track_id: int


@dataclass
class BundleAdjustmentResult:
    window_frames: list[int]
    graph: gtsam.NonlinearFactorGraph
    initial: gtsam.Values
    optimized: gtsam.Values
    track_ids: list[int]
    projection_factor_metadata: list[ProjectionFactorMetadata]
    initial_error: float
    final_error: float
    num_factors: int
    average_initial_error: float
    average_final_error: float


@dataclass
class ProjectionEvaluation:
    factor_error: float
    projection: gtsam.StereoPoint2
    left_distance_pixels: float
    right_distance_pixels: float


@dataclass
class LargestFactorDiagnostic:
    factor_index: int
    frame_id: int
    track_id: int

    measurement: gtsam.StereoPoint2

    initial: ProjectionEvaluation
    optimized: ProjectionEvaluation

@dataclass

class BundleWindowSolution:
    start_frame: int
    end_frame: int
    result: BundleAdjustmentResult
