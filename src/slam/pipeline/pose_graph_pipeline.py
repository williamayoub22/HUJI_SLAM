# solve BA windows and extract all constraints

import numpy as np

from slam.ba.gtsam_utils import make_gtsam_stereo_calibration
from slam.ba.window_selection import choose_keyframes_by_interval
from slam.ba.window_solver import solve_all_bundle_windows
from slam.config import (
    DB_PATH,
    GLOBAL_CAMERA_MATRICES_PATH,
)
from slam.io.calibration import read_stereo_calibration
from slam.pipeline.database_pipeline import load_tracking_database
from slam.pose_graph.constraints import extract_relative_pose_constraint


def load_ex6_inputs():
    db = load_tracking_database(DB_PATH)

    world_to_camera_extrinsics = np.load(
        GLOBAL_CAMERA_MATRICES_PATH,
    )

    projection_left, projection_right = read_stereo_calibration()

    calibration = make_gtsam_stereo_calibration(
        projection_left,
        projection_right,
    )

    return db, world_to_camera_extrinsics, calibration


def solve_bundle_windows_and_extract_constraints(
    db,
    world_to_camera_extrinsics,
    calibration,
    keyframe_step: int = 10,
    verbose: bool = True,
):
    """Solves all local BA windows and converts each optimized window into
    one relative keyframe constraint for the pose graph.
    """
    keyframes = choose_keyframes_by_interval(
        num_frames=len(world_to_camera_extrinsics),
        step=keyframe_step,
    )

    bundle_solutions = solve_all_bundle_windows(
        db=db,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
        keyframes=keyframes,
        verbose=verbose,
    )

    constraints = [extract_relative_pose_constraint(solution) for solution in bundle_solutions]

    return bundle_solutions, constraints
