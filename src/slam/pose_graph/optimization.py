import time
import numpy as np
import gtsam

from src.slam.pose_graph.results import PoseGraphResult


def optimize_pose_graph(
    graph: gtsam.NonlinearFactorGraph,
    initial_estimates: gtsam.Values,
    keyframe_ids: list[int],
) -> PoseGraphResult:
    print("=" * 60)
    print("Pose-graph optimization diagnostics")
    print(f"Number of factors: {graph.size()}")
    print(f"Number of values:  {initial_estimates.size()}")
    print(f"Number of keyframes: {len(keyframe_ids)}")

    t0 = time.perf_counter()
    initial_error = graph.error(initial_estimates)
    print(f"Initial error: {initial_error:.6f}")
    print(f"Initial error time: {time.perf_counter() - t0:.2f}s")

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(500)
    params.setVerbosityLM("SUMMARY")

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_estimates,
        params,
    )

    print("Starting LM optimization...")
    t0 = time.perf_counter()
    optimized_estimates = optimizer.optimize()
    print(f"Optimization time: {time.perf_counter() - t0:.2f}s")

    final_error = graph.error(optimized_estimates)
    print(f"Final error: {final_error:.6f}")

    return PoseGraphResult(
        graph=graph,
        initial_estimates=initial_estimates,
        optimized_estimates=optimized_estimates,
        keyframe_ids=keyframe_ids,
        initial_error=initial_error,
        final_error=final_error,
    )