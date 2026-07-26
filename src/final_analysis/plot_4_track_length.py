import matplotlib.pyplot as plt
from src.slam.ba.window_solver import solve_all_bundle_windows
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import pickle
from pathlib import Path
from time import perf_counter
import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import C, Q
from tqdm import tqdm
from src.slam.ba.gtsam_utils import (
    make_stereo_camera,
    pose3_from_world_to_camera_extrinsic,
    stereo_point_from_triplet,
    make_gtsam_stereo_calibration,
    stereo_image_distances,
)
from src.slam.ba.optimization import _validate_graph_keys
from src.slam.ba.results import BundleAdjustmentResult, ProjectionFactorMetadata, BundleWindowSolution
from src.slam.ba.window_selection import collect_window_tracks, bundle_windows_from_keyframes, choose_keyframes_by_motion
from src.slam.config import (
    SEQUENCE_DIR,
    CACHE_DIR,
    GT_POSES_PATH,
    GLOBAL_CAMERA_MATRICES_PATH
)
from src.slam.database.facade import SlamDatabase
from src.slam.features.detectors import extract_features
from src.slam.features.matching import match_and_filter, get_matched_points
from src.slam.geometry.ransac import ransac_pnp
from src.slam.geometry.transforms import to_homogeneous_transform
from src.slam.io.calibration import read_stereo_calibration
from src.slam.io.image_loader import read_images
from src.slam.pipeline.database_pipeline import _create_frame_data, _match_temporal_features
from src.slam.geometry.triangulation import triangulate_opencv
from src.slam.loop_closure.candidate_detection import (
    ConsensusMatcher,
    ConsensusMatchResult,
    LoopClosureCandidate,
    RelativePoseEstimate,
    detect_candidates_for_keyframe,
    refine_relative_pose_with_bundle_adjustment,
    score_candidates_for_keyframe,
    select_spread_loop_closures,
)
from src.slam.pose_graph.constraints import extract_relative_pose_constraint
from src.slam.pose_graph.graph_builder import build_pose_graph
from src.slam.pose_graph.optimization import optimize_pose_graph
from src.slam.visualization.ex7_plots import (
    plot_absolute_location_error,
    plot_consensus_match,
    plot_location_uncertainty,
    plot_pose_graph_comparisons,
    plot_pose_graphs_versions,
    positions_from_values,
)
from src.slam.pipeline.pose_graph_pipeline import solve_bundle_windows_and_extract_constraints
from src.ex8 import build_database, load_or_build_db, load_or_build_pose_graph_with_lc, load_or_build_pose_graph_no_lc, CACHE_DIR

from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_build_tracking_analysis()
    lens = data["track_lengths"]
    mean_val = lens.mean()
    plt.figure(figsize=(10, 4))
    plt.hist(lens, bins=max(2, int(lens.max())), color="teal", alpha=0.7)
    plt.axvline(mean_val, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
    plt.yscale("log")
    plt.xlabel("Track Length (frames)")
    plt.ylabel("Number of Tracks")
    plt.title("Distribution of Track Lengths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "4_track_length_histogram.png", dpi=150)
if __name__ == "__main__": main()


def _get_result_value(result, possible_names):
    """Read a value from either a dictionary or a result object."""
    for name in possible_names:
        if isinstance(result, dict) and name in result:
            return result[name]

        if hasattr(result, name):
            return getattr(result, name)

    raise AttributeError(f"Could not find any of {possible_names} in bundle-window result.")

def pose3_camera_to_world_to_extrinsic(pose):
    """Convert a GTSAM camera-to-world Pose3 into a 3x4 world-to-camera matrix.

    Bundle poses appear to represent camera poses in frame 0 because their
    translations are used directly as camera positions. Projection requires
    world-to-camera extrinsics, so the pose is inverted here.
    """
    camera_to_world = np.asarray(pose.matrix(), dtype=float)
    world_to_camera = np.linalg.inv(camera_to_world)

    return world_to_camera[:3, :]

def get_stereo_observation(manager_2d, frame_id, track_id):
    """Return the observation as (x_left, x_right, y).

    This isolates the database-specific API in one place. If manager_2d does
    not expose link_triplet directly, only this function needs to be changed.
    """
    if hasattr(manager_2d, "link_triplet"):
        return manager_2d.link_triplet(frame_id, track_id)

    # Some SlamDatabase facades expose the old TrackingDB through an
    # attribute. Keep this branch only if it matches your implementation.
    if hasattr(manager_2d, "database") and hasattr(
        manager_2d.database,
        "link_triplet",
    ):
        return manager_2d.database.link_triplet(frame_id, track_id)

    raise AttributeError(
        "Could not retrieve a stereo observation. Implement "
        "get_stereo_observation() using the manager_2d link API. "
        "It must return (x_left, x_right, y)."
    )

def compute_pnp_analysis(database):
    """
    Compute temporal PnP reprojection errors using only estimated camera poses.

    A point is triangulated in the latest camera of each track and propagated
    into earlier cameras through the composed estimated camera motions.
    """
    print("\n" + "=" * 60)
    print("PnP temporal projection-error analysis")
    print("=" * 60)

    pnp_extrinsics = np.asarray(
        np.load(GLOBAL_CAMERA_MATRICES_PATH),
        dtype=float,
    )

    estimated_world_to_camera_by_frame = {
        frame_id: to_homogeneous_transform(extrinsic)
        for frame_id, extrinsic in enumerate(pnp_extrinsics)
    }

    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)

    (
        distances,
        median_left_errors,
        median_right_errors,
        sample_counts,
    ) = compute_median_projection_errors_by_distance(
        database=database,
        world_to_camera_by_frame=(estimated_world_to_camera_by_frame),
        calibration=calibration,
        optimize_landmarks=False,
        min_track_length=2,
        max_distance=50,
    )

    return {
        "distances": distances,
        "median_left_errors": median_left_errors,
        "median_right_errors": median_right_errors,
        "sample_counts": sample_counts,
    }

def get_relative_camera_transform(
    world_to_camera_by_frame,
    source_frame,
    target_frame,
):
    """
    Return the estimated transform from source-camera coordinates to
    target-camera coordinates.

    Given:
        T_source_world: world -> source camera
        T_target_world: world -> target camera

    Then:
        T_target_source =
            T_target_world @ inverse(T_source_world)
    """
    source_world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[source_frame])
    target_world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[target_frame])

    return target_world_to_camera @ np.linalg.inv(source_world_to_camera)

def compose_camera_transform(
    world_to_camera_by_frame,
    source_frame,
    target_frame,
):
    """
    Compose all consecutive estimated camera transforms from source_frame
    to target_frame.

    The returned transform maps a point from source-camera coordinates into
    target-camera coordinates.
    """
    if source_frame == target_frame:
        return np.eye(4, dtype=float)

    step = 1 if target_frame > source_frame else -1

    current_frame = source_frame
    target_from_source = np.eye(4, dtype=float)

    while current_frame != target_frame:
        next_frame = current_frame + step

        if (
            current_frame not in world_to_camera_by_frame
            or next_frame not in world_to_camera_by_frame
        ):
            raise KeyError(
                f"Missing estimated camera pose for transition {current_frame} -> {next_frame}"
            )

        next_from_current = get_relative_camera_transform(
            world_to_camera_by_frame=world_to_camera_by_frame,
            source_frame=current_frame,
            target_frame=next_frame,
        )

        target_from_source = next_from_current @ target_from_source

        current_frame = next_frame

    return target_from_source

def load_or_build_tracking_analysis():
    import pickle
    from src.slam.config import CACHE_DIR
    path = CACHE_DIR / "tracking_analysis.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    db = load_or_build_db()
    manager_2d = db.manager_2d
    frame_ids = sorted(manager_2d.all_frames())
    track_ids = list(manager_2d.all_tracks())
    track_lengths = [len(manager_2d.frames(tid)) for tid in track_ids]
    matches_per_frame = [len(manager_2d.tracks(fid)) for fid in frame_ids]
    
    frame_connectivities = []
    for current_frame, next_frame in zip(frame_ids[:-1], frame_ids[1:]):
        current_tracks = set(manager_2d.tracks(current_frame))
        next_tracks = set(manager_2d.tracks(next_frame))
        frame_connectivities.append(len(current_tracks.intersection(next_tracks)))
        
    import numpy as np
    data = {
        "frame_ids": frame_ids,
        "track_lengths": np.asarray(track_lengths, dtype=float),
        "matches_per_frame": matches_per_frame,
        "frame_connectivities": frame_connectivities
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data
