# solve BA windows and extract all constraints

import numpy as np

from src.slam.ba.gtsam_utils import make_gtsam_stereo_calibration
from src.slam.ba.window_selection import choose_keyframes_by_motion
from src.slam.ba.window_solver import solve_all_bundle_windows
from src.slam.config import (
    DB_PATH,
    GLOBAL_CAMERA_MATRICES_PATH,
)
from src.slam.io.calibration import read_stereo_calibration
from src.slam.database.facade import SlamDatabase
from src.slam.pose_graph.constraints import extract_relative_pose_constraint


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
    slam_db: SlamDatabase,
    calibration,
    verbose: bool = True,
):
    """Solve local BA windows and convert each optimized window into
    one relative keyframe constraint for the pose graph.
    """
    poses = [slam_db.manager_poses.get_pose(i) for i in range(len(slam_db.manager_poses.get_all_poses()))]
    keyframes = choose_keyframes_by_motion(poses=np.array(poses))

    if verbose:
        print(f"Selected {len(keyframes)} motion-based keyframes.")
        print(f"First keyframes: {keyframes[:10]}")
        print(f"Last keyframes:  {keyframes[-10:]}")

    bundle_solutions = solve_all_bundle_windows(
        slam_db=slam_db,
        calibration=calibration,
        keyframes=keyframes,
        verbose=verbose,
    )

    constraints = [extract_relative_pose_constraint(solution) for solution in bundle_solutions]

    return bundle_solutions, constraints
