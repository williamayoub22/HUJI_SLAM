import gtsam
import numpy as np
from gtsam.symbol_shorthand import C

from slam.ba.gtsam_utils import (
    make_gtsam_stereo_calibration,
    pose3_from_world_to_camera_extrinsic,
)
from slam.ba.window_selection import choose_keyframes_by_interval
from slam.ba.window_solver import solve_all_bundle_windows
from slam.config import (
    DB_PATH,
    EX6_OUTPUT_DIR,
    GLOBAL_CAMERA_MATRICES_PATH,
)
from slam.io.calibration import read_stereo_calibration
from slam.pipeline.database_pipeline import load_tracking_database
from slam.pose_graph.constraints import (
    extract_relative_pose_constraint,
)


def init():
    db = load_tracking_database(DB_PATH)
    world_to_camera_extrinsics = np.load(GLOBAL_CAMERA_MATRICES_PATH)

    projection_left, projection_right = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(
        projection_left,
        projection_right,
    )

    return db, world_to_camera_extrinsics, calibration


def print_relative_constraint(constraint) -> None:
    np.set_printoptions(
        precision=6,
        suppress=True,
        linewidth=140,
    )

    covariance_eigenvalues = np.linalg.eigvalsh(constraint.covariance)

    print("=" * 60)
    print("[6.1] Relative pose constraint from first BA window")
    print("=" * 60)
    print(f"Keyframes: {constraint.start_frame} -> {constraint.end_frame}")

    print("\nRelative pose:")
    print(constraint.relative_pose)

    print("\nConditional relative covariance Cov(c_end | c_start):")
    print(constraint.covariance)

    print("\nTranslation covariance block:")
    print(constraint.covariance[3:6, 3:6])

    print("\nCovariance eigenvalues:")
    print(covariance_eigenvalues)


def verify_relative_measurement(
    first_solution,
    constraint,
) -> None:
    """Checks that c_start * delta_c equals the optimized c_end."""
    optimized = first_solution.result.optimized

    start_pose = optimized.atPose3(C(first_solution.start_frame))
    end_pose = optimized.atPose3(C(first_solution.end_frame))

    reconstructed_end_pose = start_pose.compose(constraint.relative_pose)

    if not reconstructed_end_pose.equals(end_pose, 1e-6):
        raise AssertionError(
            "Relative pose reconstruction failed. Check pose direction or composition order."
        )

    if not np.allclose(
        constraint.covariance,
        constraint.covariance.T,
        atol=1e-8,
    ):
        raise AssertionError("Relative covariance is not symmetric.")


def question6_1(
    db,
    world_to_camera_extrinsics,
    calibration,
):
    keyframes = choose_keyframes_by_interval(
        num_frames=len(world_to_camera_extrinsics),
        step=10,
    )

    solutions = solve_all_bundle_windows(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
        keyframes=keyframes,
        verbose=True,
    )

    # First BA window: required detailed analysis and console output.
    first_solution = solutions[0]
    first_constraint = extract_relative_pose_constraint(first_solution)

    verify_relative_measurement(
        first_solution,
        first_constraint,
    )

    print_relative_constraint(first_constraint)

    # Remaining BA windows: extract constraints for Ex6.2,
    # but do not print them individually.
    constraints = []

    for solution in solutions:
        try:
            constraint = extract_relative_pose_constraint(solution)
            constraints.append(constraint)

        except RuntimeError as error:
            print(
                "\n[6.1] Failed to extract covariance for window "
                f"{solution.start_frame} -> {solution.end_frame}"
            )
            print(error)
            raise

    print(
        f"\n[6.1] Extracted {len(constraints)} relative pose constraints "
        "for consecutive keyframe pairs."
    )

    return solutions, constraints


def question6_2(
    world_to_camera_extrinsics: np.ndarray,
    constraints: list,
):
    print("\n" + "=" * 60)
    print("[6.2] Global Pose Graph Optimization")
    print("=" * 60)

    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    # 1. Add Prior to anchor the first frame
    first_frame = constraints[0].start_frame
    first_pose = pose3_from_world_to_camera_extrinsic(world_to_camera_extrinsics[first_frame])

    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
    graph.add(gtsam.PriorFactorPose3(C(first_frame), first_pose, prior_noise))
    initial_estimates.insert(C(first_frame), first_pose)

    # 2. Add Relative Constraints (BetweenFactorPose3) and Chain Odometry
    for constraint in constraints:
        # Add the extracted edge to the graph
        noise_model = gtsam.noiseModel.Gaussian.Covariance(constraint.covariance)
        graph.add(
            gtsam.BetweenFactorPose3(
                C(constraint.start_frame),
                C(constraint.end_frame),
                constraint.relative_pose,
                noise_model,
            )
        )

        # Initialize the next frame by chaining the relative pose
        if not initial_estimates.exists(C(constraint.end_frame)):
            start_pose = initial_estimates.atPose3(C(constraint.start_frame))
            next_pose = start_pose.compose(constraint.relative_pose)
            initial_estimates.insert(C(constraint.end_frame), next_pose)

    # 3. Optimize the global graph
    print(f"Optimizing graph with {graph.size()} factors and {initial_estimates.size()} poses...")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
    optimized_estimates = optimizer.optimize()

    print("Optimization complete!")
    print(f"Initial Error (Pure Odometry): {graph.error(initial_estimates):.2f}")
    print(f"Final Error (Global PGO):      {graph.error(optimized_estimates):.2f}")

    # 4. Extract Trajectories
    odom_positions = np.vstack(
        [
            initial_estimates.atPose3(C(frame)).translation()
            for frame in sorted([c.start_frame for c in constraints] + [constraints[-1].end_frame])
        ]
    )

    opt_positions = np.vstack(
        [
            optimized_estimates.atPose3(C(frame)).translation()
            for frame in sorted([c.start_frame for c in constraints] + [constraints[-1].end_frame])
        ]
    )

    keyframes = sorted([c.start_frame for c in constraints] + [constraints[-1].end_frame])
    gt_extrinsics_subset = world_to_camera_extrinsics[keyframes]
    gt_positions = np.vstack(
        [
            pose3_from_world_to_camera_extrinsic(gt_extrinsics_subset[i]).translation()
            for i in range(len(gt_extrinsics_subset))
        ]
    )

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    # Ensure output directory exists
    output_dir = EX6_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    def plot_custom(estimated, gt, title, output_path, marginals=None, keys=None):
        plt.figure(figsize=(10, 8))
        plt.plot(estimated[:, 0], estimated[:, 2], label="Pose Graph", color="blue", linewidth=2)
        plt.plot(gt[:, 0], gt[:, 2], label="Ground Truth (Ex 3)", color="orange", linestyle="--")

        # Plot Covariance Ellipses if provided
        if marginals is not None and keys is not None:
            ax = plt.gca()
            for i, key in enumerate(keys):
                # Only plot every 10th ellipse so it doesn't become a solid black blob
                if i % 10 == 0:
                    cov = marginals.marginalCovariance(key)
                    # Extract the translation covariance block (bottom right 3x3 of the 6x6)
                    # For Pose3, GTSAM ordering is [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
                    trans_cov = cov[3:6, 3:6]
                    # We want X and Z axes (index 0 and 2)
                    xz_cov = np.array(
                        [[trans_cov[0, 0], trans_cov[0, 2]], [trans_cov[2, 0], trans_cov[2, 2]]]
                    )

                    eigvals, eigvecs = np.linalg.eigh(xz_cov)
                    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

                    # Scale by 3 sigma (99.7% confidence)
                    width, height = 2 * 3 * np.sqrt(np.maximum(eigvals, 1e-9))
                    ellip = Ellipse(
                        xy=(estimated[i, 0], estimated[i, 2]),
                        width=width,
                        height=height,
                        angle=angle,
                        edgecolor="red",
                        facecolor="none",
                        alpha=0.5,
                    )
                    ax.add_patch(ellip)

            # Dummy line for legend
            plt.plot([], [], color="red", label="3-Sigma Covariance")

        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        print(f"Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=200)
        plt.close()

    print("\nPlotting 1: Initial Poses (Odometry) vs Ground Truth")
    plot_custom(
        odom_positions,
        gt_positions,
        "Ex 6.2: Initial Poses (Odometry Chaining)",
        output_dir / "q6_2_initial_poses.png",
    )

    print("Plotting 2: Optimized Locations (Without Covariances)")
    plot_custom(
        opt_positions,
        gt_positions,
        "Ex 6.2: Optimized Locations (No Covariances)",
        output_dir / "q6_2_optimized_poses.png",
    )

    print("Plotting 3: Optimized Locations WITH Final Marginal Covariances")
    print("Calculating global marginals (this might take a few seconds)...")
    global_marginals = gtsam.Marginals(graph, optimized_estimates)
    keys = [C(f) for f in keyframes]
    plot_custom(
        opt_positions,
        gt_positions,
        "Ex 6.2: Optimized Locations WITH Marginal Covariances",
        output_dir / "q6_2_optimized_with_covariances.png",
        global_marginals,
        keys,
    )


def main():
    db, world_to_camera_extrinsics, calibration = init()

    solutions, constraints = question6_1(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
    )

    question6_2(
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        constraints=constraints,
    )


if __name__ == "__main__":
    main()
