import gtsam

from slam.pose_graph.results import PoseGraphResult


def optimize_pose_graph(
    graph: gtsam.NonlinearFactorGraph,
    initial_estimates: gtsam.Values,
    keyframe_ids: list[int],
) -> PoseGraphResult:
    """Optimizes a pose graph and stores initial/final factor-graph errors."""
    initial_error = graph.error(initial_estimates)

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_estimates,
    )
    optimized_estimates = optimizer.optimize()

    final_error = graph.error(optimized_estimates)

    return PoseGraphResult(
        graph=graph,
        initial_estimates=initial_estimates,
        optimized_estimates=optimized_estimates,
        keyframe_ids=keyframe_ids,
        initial_error=initial_error,
        final_error=final_error,
    )


def compute_pose_graph_marginals(
    pose_graph_result: PoseGraphResult,
) -> gtsam.Marginals:
    return gtsam.Marginals(
        pose_graph_result.graph,
        pose_graph_result.optimized_estimates,
    )
