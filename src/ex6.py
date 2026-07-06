from slam.analysis.pose_graph_diagnostics import (
    format_pose_graph_report,
    format_relative_constraint_report,
)
from slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic
from slam.config import EX6_OUTPUT_DIR
from slam.pipeline.pose_graph_pipeline import (
    load_ex6_inputs,
    solve_bundle_windows_and_extract_constraints,
)

from slam.pose_graph.graph_builder import build_pose_graph
from slam.pose_graph.optimization import (
    compute_pose_graph_marginals,
    optimize_pose_graph,
)
from slam.visualization.ex6_plots import (
    plot_bundle_window_trajectory_with_covariances,
    plot_pose_graph_top_down,
    positions_from_values,
)


def q_1(
    db,
    world_to_camera_extrinsics,
    calibration,
):
    """Section 6.1:
    1. Solve all BA windows from Ex5.
    2. Extract one relative pose constraint and conditional covariance
       for every consecutive keyframe pair.
    3. Print the first constraint.
    4. Plot all poses in the first BA window with their marginal
       covariance visualization.

    Returns:
        constraints: relative constraints for all consecutive keyframe pairs.
    """
    bundle_solutions, constraints = solve_bundle_windows_and_extract_constraints(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
        keyframe_step=10,
        verbose=True,
    )

    first_solution = bundle_solutions[0]
    first_constraint = constraints[0]

    # verify_relative_measurement(
    #     bundle_solution=first_solution,
    #     constraint=first_constraint,
    # )

    print(
        format_relative_constraint_report(
            first_constraint,
        )
    )

    plot_bundle_window_trajectory_with_covariances(
        graph=first_solution.result.graph,
        optimized_values=first_solution.result.optimized,
        output_path=EX6_OUTPUT_DIR / "q6_1_first_bundle_covariances.png",
    )

    print(
        f"\n[6.1] Extracted {len(constraints)} relative-pose "
        "constraints for consecutive keyframe pairs."
    )

    return constraints


def q_2(
    world_to_camera_extrinsics,
    constraints,
):
    """Section 6.2:
    1. Build the keyframe pose graph.
    2. Initialize it by chaining the relative-pose constraints.
    3. Optimize it.
    4. Plot initial, optimized, and covariance-aware optimized poses.
    """
    first_frame = constraints[0].start_frame
    first_pose = pose3_from_world_to_camera_extrinsic(
        world_to_camera_extrinsics[first_frame],
    )

    graph, initial_estimates, keyframe_ids = build_pose_graph(
        constraints=constraints,
        first_pose=first_pose,
    )

    pose_graph_result = optimize_pose_graph(
        graph=graph,
        initial_estimates=initial_estimates,
        keyframe_ids=keyframe_ids,
    )

    print(
        format_pose_graph_report(
            initial_error=pose_graph_result.initial_error,
            final_error=pose_graph_result.final_error,
            num_factors=pose_graph_result.graph.size(),
            num_poses=pose_graph_result.initial_estimates.size(),
        )
    )

    initial_positions = positions_from_values(
        values=pose_graph_result.initial_estimates,
        frame_ids=keyframe_ids,
    )

    optimized_positions = positions_from_values(
        values=pose_graph_result.optimized_estimates,
        frame_ids=keyframe_ids,
    )

    plot_pose_graph_top_down(
        estimated_positions=initial_positions,
        output_path=EX6_OUTPUT_DIR / "q6_2_initial_poses.png",
        title="Ex 6.2: Initial Pose-Graph Estimate",
    )

    plot_pose_graph_top_down(
        estimated_positions=optimized_positions,
        output_path=EX6_OUTPUT_DIR / "q6_2_optimized_poses.png",
        title="Ex 6.2: Optimized Pose-Graph Estimate",
    )

    marginals = compute_pose_graph_marginals(
        pose_graph_result,
    )

    plot_pose_graph_top_down(
        estimated_positions=optimized_positions,
        output_path=EX6_OUTPUT_DIR / "q6_2_optimized_with_covariances.png",
        title="Ex 6.2: Optimized Locations WITH Marginal Covariances",
        marginals=marginals,
        frame_ids=keyframe_ids,
    )

    return pose_graph_result


def main() -> None:
    db, world_to_camera_extrinsics, calibration = load_ex6_inputs()

    constraints = q_1(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
    )

    q_2(
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        constraints=constraints,
    )


if __name__ == "__main__":
    main()
