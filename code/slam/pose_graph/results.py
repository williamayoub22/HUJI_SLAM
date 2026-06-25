from dataclasses import dataclass

import gtsam


@dataclass(frozen=True)
class PoseGraphResult:
    graph: gtsam.NonlinearFactorGraph
    initial_estimates: gtsam.Values
    optimized_estimates: gtsam.Values
    keyframe_ids: list[int]
    initial_error: float
    final_error: float
