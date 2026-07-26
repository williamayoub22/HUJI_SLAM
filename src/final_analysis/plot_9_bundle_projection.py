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
    data = load_or_build_projection_data()["ba_projection_data"]
    plt.figure(figsize=(10, 5))
    plt.plot(data["distances"], data["median_left_errors"], label="Left", marker="o", markersize=3)
    plt.plot(data["distances"], data["median_right_errors"], label="Right", marker="o", markersize=3)
    plt.xlabel("Distance from Triangulation [frames]")
    plt.ylabel("Median Projection Error [px]")
    plt.title("Bundle Adjustment Projection Error vs Distance")
    plt.xlim(0, 50)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "9_bundle_projection_error_vs_distance.png", dpi=150)
if __name__ == "__main__": main()


def compute_median_projection_errors_by_distance(
    database,
    world_to_camera_by_frame,
    calibration,
    *,
    optimize_landmarks,
    min_track_length=2,
    max_distance=50,
):
    """
    Compute median stereo reprojection errors by temporal distance.

    PnP mode:
        1. Backproject the latest stereo observation into the latest camera's
           coordinate system.
        2. Propagate that 3D point into each earlier camera by composing the
           consecutive estimated camera transformations.
        3. Project the propagated point into that frame.
        4. Compare the predicted pixels with the measured stereo observation.

    Bundle-adjustment mode:
        Optimize one world-space landmark using all selected observations,
        while keeping the supplied camera poses fixed.
    """
    manager_2d = database.manager_2d

    left_errors_by_distance = defaultdict(list)
    right_errors_by_distance = defaultdict(list)

    used_tracks = 0
    rejected_tracks = 0

    # An identity-pose camera projects points that are already expressed
    # in the current camera coordinate system.
    local_stereo_camera = make_stereo_camera(
        np.eye(4, dtype=float),
        calibration,
    )

    for track_id in manager_2d.all_tracks():
        track_frames = sorted(manager_2d.frames(track_id))

        usable_frames = [
            frame_id for frame_id in track_frames if frame_id in world_to_camera_by_frame
        ]

        if len(usable_frames) < min_track_length:
            continue

        triangulation_frame = usable_frames[-1]

        evaluation_frames = [
            frame_id
            for frame_id in usable_frames
            if 0 <= triangulation_frame - frame_id <= max_distance
        ]

        if len(evaluation_frames) < min_track_length:
            continue

        try:
            if optimize_landmarks:
                landmark_world = optimize_track_landmark(
                    manager_2d=manager_2d,
                    track_id=track_id,
                    frame_ids=evaluation_frames,
                    world_to_camera_by_frame=world_to_camera_by_frame,
                    calibration=calibration,
                )

                landmark_world = np.asarray(
                    landmark_world,
                    dtype=float,
                ).reshape(3)

                if not np.all(np.isfinite(landmark_world)):
                    raise ValueError("Optimized landmark contains non-finite values.")

            else:
                # Backproject using an identity camera. The result is expressed
                # in triangulation_frame camera coordinates, not world space.
                triangulation_measurement = stereo_point_from_triplet(
                    get_stereo_observation(
                        manager_2d,
                        triangulation_frame,
                        track_id,
                    )
                )

                landmark_in_triangulation_camera = np.asarray(
                    local_stereo_camera.backproject(triangulation_measurement),
                    dtype=float,
                ).reshape(3)

                if not np.all(np.isfinite(landmark_in_triangulation_camera)):
                    raise ValueError("Triangulated landmark contains non-finite values.")

        except (
            KeyError,
            IndexError,
            ValueError,
            RuntimeError,
            Exception,
        ):
            rejected_tracks += 1
            continue

        track_contributed = False

        for frame_id in evaluation_frames:
            distance = triangulation_frame - frame_id

            try:
                if optimize_landmarks:
                    frame_extrinsic = to_homogeneous_transform(world_to_camera_by_frame[frame_id])

                    frame_camera = make_stereo_camera(
                        frame_extrinsic,
                        calibration,
                    )

                    projection = frame_camera.project(gtsam.Point3(landmark_world))

                else:
                    # Map the point from the triangulation camera to the
                    # requested evaluation camera by composing estimated
                    # consecutive camera motions.
                    frame_from_triangulation = compose_camera_transform(
                        world_to_camera_by_frame=(world_to_camera_by_frame),
                        source_frame=triangulation_frame,
                        target_frame=frame_id,
                    )

                    landmark_homogeneous = np.append(
                        landmark_in_triangulation_camera,
                        1.0,
                    )

                    landmark_in_frame_camera = (frame_from_triangulation @ landmark_homogeneous)[:3]

                    if not np.all(np.isfinite(landmark_in_frame_camera)):
                        continue

                    # The point is already in frame_id camera coordinates.
                    projection = local_stereo_camera.project(gtsam.Point3(landmark_in_frame_camera))

                measurement = stereo_point_from_triplet(
                    get_stereo_observation(
                        manager_2d,
                        frame_id,
                        track_id,
                    )
                )

                left_error, right_error = stereo_image_distances(
                    measurement=measurement,
                    projection=projection,
                )

            except (
                KeyError,
                IndexError,
                ValueError,
                RuntimeError,
                Exception,
            ):
                continue

            left_errors_by_distance[distance].append(float(left_error))
            right_errors_by_distance[distance].append(float(right_error))

            track_contributed = True

        if track_contributed:
            used_tracks += 1
        else:
            rejected_tracks += 1

    distances = np.asarray(
        sorted(set(left_errors_by_distance) & set(right_errors_by_distance)),
        dtype=int,
    )

    median_left_errors = np.asarray(
        [np.median(left_errors_by_distance[distance]) for distance in distances],
        dtype=float,
    )

    median_right_errors = np.asarray(
        [np.median(right_errors_by_distance[distance]) for distance in distances],
        dtype=float,
    )

    sample_counts = np.asarray(
        [
            min(
                len(left_errors_by_distance[distance]),
                len(right_errors_by_distance[distance]),
            )
            for distance in distances
        ],
        dtype=int,
    )

    print(f"Used tracks:        {used_tracks}")
    print(f"Rejected tracks:    {rejected_tracks}")
    print(f"Computed distances: {len(distances)}")

    return (
        distances,
        median_left_errors,
        median_right_errors,
        sample_counts,
    )

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

def load_or_build_projection_data():
    import pickle
    path = CACHE_DIR / "projection_data.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    db = load_or_build_db()
    pnp_projection_data = compute_pnp_analysis(db)
    ba_projection_data = compute_bundle_adjustment_analysis(db)
    
    data = {
        "pnp_projection_data": pnp_projection_data,
        "ba_projection_data": ba_projection_data
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data

def optimize_track_landmark(
    manager_2d,
    track_id,
    frame_ids,
    world_to_camera_by_frame,
    calibration,
    measurement_sigma_pixels=1.0,
):
    """Optimize one landmark using all stereo observations in one track.

    Camera poses are fixed using tight Pose3 priors. Only the landmark is
    meaningfully adjusted by the optimization.
    """
    if len(frame_ids) < 2:
        raise ValueError("At least two observations are required.")

    triangulation_frame = frame_ids[-1]

    triangulation_extrinsic = to_homogeneous_transform(
        world_to_camera_by_frame[triangulation_frame]
    )

    triangulation_camera = make_stereo_camera(
        triangulation_extrinsic,
        calibration,
    )

    triangulation_measurement = stereo_point_from_triplet(
        get_stereo_observation(
            manager_2d,
            triangulation_frame,
            track_id,
        )
    )

    initial_landmark = triangulation_camera.backproject(triangulation_measurement)

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    landmark_key = Q(int(track_id))

    initial_values.insert(
        landmark_key,
        gtsam.Point3(np.asarray(initial_landmark, dtype=float)),
    )

    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        3,
        measurement_sigma_pixels,
    )

    # Very tight priors keep all PnP poses fixed.
    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-7, dtype=float))

    for frame_id in frame_ids:
        pose_key = C(int(frame_id))

        world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[frame_id])

        camera_pose = pose3_from_world_to_camera_extrinsic(world_to_camera)

        initial_values.insert(pose_key, camera_pose)

        graph.add(
            gtsam.PriorFactorPose3(
                pose_key,
                camera_pose,
                pose_prior_noise,
            )
        )

        measurement = stereo_point_from_triplet(
            get_stereo_observation(
                manager_2d,
                frame_id,
                track_id,
            )
        )

        graph.add(
            gtsam.GenericStereoFactor3D(
                measurement,
                measurement_noise,
                pose_key,
                landmark_key,
                calibration,
            )
        )

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_values,
    )

    optimized_values = optimizer.optimize()

    optimized_landmark = np.asarray(
        optimized_values.atPoint3(landmark_key),
        dtype=float,
    ).reshape(3)

    return optimized_landmark
