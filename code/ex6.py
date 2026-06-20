import numpy as np
from gtsam.symbol_shorthand import C

from slam.ba.gtsam_utils import make_gtsam_stereo_calibration
from slam.ba.window_selection import choose_keyframes_by_interval
from slam.ba.window_solver import solve_all_bundle_windows
from slam.config import (
    DB_PATH,
    GLOBAL_CAMERA_MATRICES_PATH,
)
from slam.io.calibration import read_stereo_calibration
from slam.pipeline.database_pipeline import load_tracking_database
from slam.pose_graph.constraints import (
    extract_relative_pose_constraint,
)


def init():
    db = load_tracking_database(DB_PATH)
    world_to_camera_extrinsics = np.load(
        GLOBAL_CAMERA_MATRICES_PATH
    )

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

    covariance_eigenvalues = np.linalg.eigvalsh(
        constraint.covariance
    )

    print("=" * 60)
    print("[6.1] Relative pose constraint from first BA window")
    print("=" * 60)
    print(
        f"Keyframes: {constraint.start_frame} "
        f"-> {constraint.end_frame}"
    )

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

    reconstructed_end_pose = start_pose.compose(
        constraint.relative_pose
    )

    if not reconstructed_end_pose.equals(end_pose, 1e-6):
        raise AssertionError(
            "Relative pose reconstruction failed. Check pose direction "
            "or composition order."
        )

    if not np.allclose(
        constraint.covariance,
        constraint.covariance.T,
        atol=1e-8,
    ):
        raise AssertionError(
            "Relative covariance is not symmetric."
        )


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


def main():
    db, world_to_camera_extrinsics, calibration = init()

    question6_1(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
    )


if __name__ == "__main__":
    main()