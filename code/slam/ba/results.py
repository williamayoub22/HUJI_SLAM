"""Data containers produced by local bundle-adjustment construction and solving."""

from dataclasses import dataclass

import gtsam


@dataclass(frozen=True)
class ProjectionFactorMetadata:
    """Identify one stereo projection factor in a local BA graph.

    Attributes:
        factor_index: Index of the factor in the nonlinear factor graph.
        frame_id: Frame associated with the camera variable.
        track_id: Track associated with the landmark variable.
    """

    factor_index: int
    frame_id: int
    track_id: int


@dataclass
class BundleAdjustmentResult:
    """Store the graph, estimates, and error statistics of one BA window.

    Attributes:
        window_frames: Frame IDs optimized in this local window.
        graph: Constructed nonlinear factor graph.
        initial: Initial pose and landmark estimates.
        optimized: Estimates returned by the optimizer.
        track_ids: Track IDs represented as landmarks in the graph.
        projection_factor_metadata: Metadata for stereo projection factors.
        initial_error: Total graph error before optimization.
        final_error: Total graph error after optimization.
        num_factors: Number of factors in the graph.
        average_initial_error: Initial graph error divided by factor count.
        average_final_error: Final graph error divided by factor count.
    """

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


@dataclass(frozen=True)
class BundleWindowSolution:
    """Associate a solved local BA result with its keyframe interval.

    Attributes:
        start_frame: First keyframe of the window.
        end_frame: Last keyframe of the window.
        result: Optimization result for the window.
    """

    start_frame: int
    end_frame: int
    result: BundleAdjustmentResult
