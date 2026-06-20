# from pathlib import Path

import cv2
import numpy as np
from gtsam.symbol_shorthand import C

from slam.analysis.bundle_diagnostics import analyze_largest_initial_projection_factor
from slam.ba.gtsam_utils import make_gtsam_stereo_calibration
from slam.ba.window_composition import (
    collect_landmarks_in_frame0,
    compose_keyframe_poses_in_frame0,
)
from slam.ba.optimization import optimize_bundle_window
from slam.analysis.track_reprojection import analyze_track_reprojection
from slam.ba.results import BundleWindowSolution
from slam.ba.window_solver import solve_all_bundle_windows
from slam.ba.window_selection import (
    choose_keyframes_by_interval,
    bundle_windows_from_keyframes,
)
from slam.config import (
    DB_PATH,
    EX5_OUTPUT_DIR,
    GLOBAL_CAMERA_MATRICES_PATH,
    GT_POSES_PATH,
    LEFT_IMAGES_DIR,
    RIGHT_IMAGES_DIR,
)
from slam.io.calibration import read_stereo_calibration
from slam.pipeline.database_pipeline import load_tracking_database
from slam.visualization.ex5_plots import (
    plot_factor_errors_q5_1,
    plot_keyframe_localization_error,
    plot_keyframes_and_landmarks_top_down,
    plot_largest_factor_diagnostic,
    plot_reprojection_errors_q5_1,
)
from slam.geometry.transforms import (
    camera_centers_from_world_to_camera_extrinsics,
)

from slam.visualization.trajectory import (
    plot_bundle_scene_3d,
    plot_bundle_scene_top_down,
)


def print_error_stats(result, reprojection_plot_path, factor_plot_path):
    print("=" * 60)
    print("[5.1] Track reprojection analysis")
    print("=" * 60)
    print(f"Selected track: {result.track_id}")
    print(f"Track length: {len(result.frame_ids)}")
    print(f"First frame: {result.frame_ids[0]}")
    print(f"Last frame: {result.frame_ids[-1]}")
    print(f"Triangulated world landmark: {np.asarray(result.landmark_world)}")
    print("Measurement covariance:")
    print(result.covariance)
    print(f"Saved reprojection plot to: {reprojection_plot_path}")
    print(f"Saved factor error plot to: {factor_plot_path}")


def question5_1(
    db,
    world_to_camera_extrinsics,
    calibration,
) -> None:
    """Run Exercise 5.1 analysis, plots, and console reporting."""
    result = analyze_track_reprojection(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
        min_track_length=10,
        seed=1,
        sigma_pixels=1.0,
    )

    reprojection_plot_path = EX5_OUTPUT_DIR / "q5_1_reprojection_error.png"
    factor_plot_path = EX5_OUTPUT_DIR / "q5_1_factor_error.png"

    plot_reprojection_errors_q5_1(
        frame_ids=result.frame_ids,
        reprojection_errors=result.reprojection_errors,
        track_id=result.track_id,
        output_path=reprojection_plot_path,
        use_track_index=True,
    )

    plot_factor_errors_q5_1(
        frame_ids=result.frame_ids,
        factor_errors=result.factor_errors,
        track_id=result.track_id,
        output_path=factor_plot_path,
        use_track_index=True,
    )

    print_error_stats(result, reprojection_plot_path, factor_plot_path)


def _print_bundle_summary(keyframes, window_frames, result):
    print("=" * 60)
    print("[5.3] First bundle window")
    print("=" * 60)
    print(f"Keyframes: {keyframes[0]} -> {keyframes[1]}")
    print(f"Window frames: {window_frames[0]} to {window_frames[-1]}")
    print(f"Number of poses: {len(window_frames)}")
    print(f"Number of landmarks: {len(result.track_ids)}")
    print(f"Number of factors: {result.num_factors}")
    print(f"Initial total graph error: {result.initial_error:.6f}")
    print(f"Final total graph error: {result.final_error:.6f}")
    print(f"Initial average factor error: {result.average_initial_error:.6f}")
    print(f"Final average factor error: {result.average_final_error:.6f}")


def question5_3(db, world_to_camera_extrinsics, calibration):
    """
    Exercise 5.3:
    Runs local bundle adjustment on the first keyframe window.
    """
    keyframes = choose_keyframes_by_interval(
        num_frames=len(world_to_camera_extrinsics),
        step=10,
    )

    window_frames = bundle_windows_from_keyframes(keyframes)[0]

    result = optimize_bundle_window(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
        window_frames=window_frames,
    )

    _print_bundle_summary(keyframes, window_frames, result)

    diagnostic = analyze_largest_initial_projection_factor(
        db=db,
        calibration=calibration,
        result=result,
    )

    frame_id = diagnostic.frame_id

    left_image_path = LEFT_IMAGES_DIR / f"{frame_id:06d}.png"
    right_image_path = RIGHT_IMAGES_DIR / f"{frame_id:06d}.png"

    left_image = cv2.imread(str(left_image_path))
    right_image = cv2.imread(str(right_image_path))

    if left_image is None or right_image is None:
        raise FileNotFoundError(
            f"Could not load the stereo pair for frame {frame_id}: "
            f"{left_image_path}, {right_image_path}"
        )

    print("=" * 60)
    print("[5.3] Largest initial-error projection factor")
    print("=" * 60)
    print(f"Factor index: {diagnostic.factor_index}")
    print(f"Frame c: {diagnostic.frame_id}")
    print(f"Landmark q: {diagnostic.track_id}")

    print(f"Initial factor error: {diagnostic.initial.factor_error:.6f}")
    print(
        "Initial distances: "
        f"left={diagnostic.initial.left_distance_pixels:.3f}px, "
        f"right={diagnostic.initial.right_distance_pixels:.3f}px"
    )

    print(f"Final factor error: {diagnostic.optimized.factor_error:.6f}")
    print(
        "Final distances: "
        f"left={diagnostic.optimized.left_distance_pixels:.3f}px, "
        f"right={diagnostic.optimized.right_distance_pixels:.3f}px"
    )

    largest_factor_plot_path = EX5_OUTPUT_DIR / "q5_3_largest_initial_factor.png"

    plot_largest_factor_diagnostic(
        diagnostic=diagnostic,
        left_image=left_image,
        right_image=right_image,
        output_path=largest_factor_plot_path,
    )

    print(f"Saved largest-factor diagnostic to: {largest_factor_plot_path}")

    trajectory_plot_path = EX5_OUTPUT_DIR / "bundle_scene_3d.png"
    top_down_plot_path = EX5_OUTPUT_DIR / "bundle_scene_top_down.png"

    plot_bundle_scene_3d(result, trajectory_plot_path)
    plot_bundle_scene_top_down(result, top_down_plot_path)

    print(f"Saved 3D bundle scene to: {trajectory_plot_path}")
    print(f"Saved top-down bundle scene to: {top_down_plot_path}")

    return result


def print_last_window_anchor_diagnostics(
    last_solution: BundleWindowSolution,
) -> None:
    result = last_solution.result
    first_frame = last_solution.start_frame

    first_pose = result.optimized.atPose3(C(first_frame))
    first_position = np.asarray(first_pose.translation(), dtype=float)

    anchor_factor = result.graph.at(0)
    anchor_final_error = anchor_factor.error(result.optimized)

    print("=" * 60)
    print("[5.4] Last bundle window diagnostics")
    print("=" * 60)
    print(f"Last window: {last_solution.start_frame} -> {last_solution.end_frame}")
    print(f"Optimized position of its first frame: {first_position}")
    print(f"Anchoring factor final error: {anchor_final_error:.12f}")


def question5_4(db, world_to_camera_extrinsics, calibration, gt_positions_all):
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

    print(f"[5.4] Solved {len(solutions)} bundle windows.")

    print_last_window_anchor_diagnostics(solutions[-1])

    keyframe_poses_in_frame0, _ = compose_keyframe_poses_in_frame0(solutions)

    landmarks_in_frame0 = collect_landmarks_in_frame0(
        solutions=solutions,
        keyframe_poses_in_frame0=keyframe_poses_in_frame0,
    )

    estimated_positions = np.vstack(
        [
            np.asarray(
                keyframe_poses_in_frame0[frame_id].translation(),
                dtype=float,
            ).reshape(3)
            for frame_id in keyframes
        ]
    )

    gt_keyframe_positions = gt_positions_all[keyframes]

    scene_path = EX5_OUTPUT_DIR / "q5_4_keyframes_and_landmarks_top_down.png"
    error_path = EX5_OUTPUT_DIR / "q5_4_keyframe_localization_error.png"

    plot_keyframes_and_landmarks_top_down(
        keyframe_ids=keyframes,
        global_keyframe_poses=keyframe_poses_in_frame0,
        landmarks_global=landmarks_in_frame0,
        gt_positions=gt_keyframe_positions,
        output_path=scene_path,
    )

    localization_errors = plot_keyframe_localization_error(
        keyframe_ids=keyframes,
        estimated_positions=estimated_positions,
        gt_positions=gt_keyframe_positions,
        output_path=error_path,
    )

    print(f"Saved 5.4 top-down scene to: {scene_path}")
    print(f"Saved localization-error plot to: {error_path}")
    print(f"Mean keyframe localization error: {localization_errors.mean():.3f} m")

    return solutions, keyframe_poses_in_frame0


def init():
    EX5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db = load_tracking_database(DB_PATH)
    world_to_camera_extrinsics = np.load(GLOBAL_CAMERA_MATRICES_PATH)

    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)

    gt_world_to_camera_extrinsics = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    gt_positions_all = camera_centers_from_world_to_camera_extrinsics(
        gt_world_to_camera_extrinsics
    )

    return calibration, db, world_to_camera_extrinsics, gt_positions_all


def main():
    calibration, db, world_to_camera_extrinsics, gt_positions_all = init()

    question5_1(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
    )
    question5_3(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
    )

    question5_4(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
        gt_positions_all=gt_positions_all,
    )


if __name__ == "__main__":
    main()
