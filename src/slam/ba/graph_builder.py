"""Construct local stereo bundle-adjustment factor graphs.

1) query the pre-triangulated global 3D points from our manager_3d instead of relying on gtsam's backproject.
   (maybe this is useless since we just pattern filter after finding matches) just another extra test.

2) we build the single prior constraints.

3) our goal is to minimize P(Qi,Cj|Zij) ∝ P(Zij| Qi,Cj) which is the same as minimizing the reprojection error.
   so we add stereo factors as follows:
        - add symbols for Q,C these represent the graph nodes.
        - add edge by passing
            * Q and C symbols,
            * the calibration for projection
            * measurement (from feature matching stored in db.link_triplet as x_left,x_right,y)
            * the noise model

4) use triangulation and poses

"""

import gtsam
import numpy as np

from .. import config
from gtsam.symbol_shorthand import C, Q
from src.slam.database.facade import SlamDatabase

from ..geometry.transforms import (
    relative_world_to_camera_extrinsic,
    to_homogeneous_transform,
)
from .gtsam_utils import (
    make_stereo_camera,
    pose3_from_world_to_camera_extrinsic,
    stereo_point_from_triplet,
)
from .results import ProjectionFactorMetadata
from .window_selection import collect_window_tracks


def build_local_bundle_graph(
    slam_db: SlamDatabase,
    calibration: gtsam.Cal3_S2Stereo,
    window_frames: list[int],
    min_track_observations: int = 2,
    measurement_sigma_pixels: float = config.MEASUREMENT_SIGMA_PIXELS,
) -> tuple[
    gtsam.NonlinearFactorGraph,
    gtsam.Values,
    list[int],
    list[ProjectionFactorMetadata],
]:
    """Build a local stereo bundle-adjustment graph for one frame window.

    Poses and landmarks are expressed in the coordinate system of the first
    window frame. The first camera pose is anchored with a prior to remove
    gauge freedom.

    Args:
        slam_db: Unified SLAM database containing 2D tracks, 3D points, and poses.
        calibration: Stereo camera calibration.
        window_frames: Consecutive frame IDs included in the local window.
        min_track_observations: Minimum observations required for a track.
        measurement_sigma_pixels: Isotropic stereo-measurement noise standard
            deviation in pixels.

    Returns:
        The factor graph, initial values, inserted landmark track IDs, and
        metadata for every stereo projection factor.

    Raises:
        ValueError: If the window is empty or configuration values are invalid.
        RuntimeError: If no valid landmarks can be initialized in the window.
    """
    if not window_frames:
        raise ValueError("Bundle-adjustment window must contain at least one frame.")

    if min_track_observations < 2:
        raise ValueError("min_track_observations must be at least 2.")

    if measurement_sigma_pixels <= 0:
        raise ValueError("measurement_sigma_pixels must be positive.")

    # init the factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # find the first frame from the current ba window
    first_frame = window_frames[0]
    
    # from ex3. here we select the first R|t matrix for the first
    # frame in the bundle
    T_world_to_first = slam_db.manager_poses.get_pose(first_frame)

    # this creates a sigma^-1 matrix of eye(3).
    # in model where zij = project(Q,C) + w, this is cov of w
    # we used measurement_sigma_pixels=1
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        3,
        measurement_sigma_pixels,
    )

    # we want to snap first camera to 0,0,0. to do this we add a constraint.
    # we still need to add a constraint in a probability system.
    # so we give it small sigma values. this makes the cov values small. meaning we are certain
    # that we should snap to 0,0,0.
    # note: since we divided to windows. we are snapping to pnp intial guess for first frame.
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-3, dtype=float))

    # find all the tracks in the current chosen window. dict[track_id, list[frame_id]]
    tracks_to_frames = collect_window_tracks(
        db=slam_db.manager_2d,
        window_frames=window_frames,
        min_observations=min_track_observations,
    )

    # Construct candidate cameras before choosing which poses enter the graph.
    # The 3D landmarks will be transformed to the first-frame coordinate system.
    stereo_cameras: dict[int, gtsam.StereoCamera] = {}
    poses: dict[int, gtsam.Pose3] = {}
    for frame_id in window_frames:
        T_world_to_frame = slam_db.manager_poses.get_pose(frame_id)

        # in ex3 we found T with respect to the world(cam at frame 0).
        # find T matrices with respect to the first frame in the bundle window
        T_first_to_frame = relative_world_to_camera_extrinsic(
            world_to_reference=T_world_to_first,
            world_to_frame=T_world_to_frame,
        )

        # our T(extrinsic matrix): maps from the world to the camera.
        # gtsam T(pose): works by mapping from the camera to the world.
        # we need to invert.
        poses[frame_id] = pose3_from_world_to_camera_extrinsic(T_first_to_frame)

        # create gtsam cam object.
        stereo_cameras[frame_id] = make_stereo_camera(
            T_first_to_frame,
            calibration,
        )

    # Retain only tracks whose global 3D point can be retrieved and transformed to the local frame.
    initialized_landmarks: dict[int, gtsam.Point3] = {}
    valid_tracks_to_frames: dict[int, list[int]] = {}

    for track_id, track_frames in tracks_to_frames.items():
        # Retrieve the pre-triangulated global 3D point from the SLAM database.
        # This point was triangulated using all track observations across the entire history,
        # ensuring the lowest possible spatial error compared to triangulating from just two frames.
        landmark_world = slam_db.manager_3d.get_point(track_id)
        if landmark_world is None:
            continue

        # Transform landmark from world coordinates to the local first-frame coordinate system
        landmark_4d = np.append(landmark_world, 1.0)
        T_w_to_f_homo = to_homogeneous_transform(T_world_to_first)
        landmark_first_frame = (T_w_to_f_homo @ landmark_4d)[:3]

        # Store the transformed point to be used as the initial guess in the local BA graph
        initialized_landmarks[track_id] = gtsam.Point3(landmark_first_frame)
        valid_tracks_to_frames[track_id] = track_frames

    if not valid_tracks_to_frames:
        raise RuntimeError("No valid landmarks could be initialized in this window.")

    # build a set that only includes points that were successfully retrieved from the global map
    active_frames = {first_frame}
    for track_frames in valid_tracks_to_frames.values():
        active_frames.update(track_frames)

    # Insert only poses that are anchored or referenced by projection factors.
    for frame_id in window_frames:
        if frame_id in active_frames:
            initial.insert(C(frame_id), poses[frame_id])

    # Lock the first frame of the bundle window to the origin (Identity pose).
    # Since all poses in the window were computed relative to the first frame, poses[first_frame]
    # is exactly the Identity matrix. This anchors the graph and removes 6-DOF gauge freedom.
    graph.add(
        gtsam.PriorFactorPose3(
            C(first_frame),
            poses[first_frame],
            prior_noise,
        )
    )

    inserted_landmarks: list[int] = []
    projection_factor_metadata: list[ProjectionFactorMetadata] = []

    for track_id, track_frames in valid_tracks_to_frames.items():
        initial.insert(Q(track_id), initialized_landmarks[track_id])
        inserted_landmarks.append(track_id)

        for frame_id in track_frames:
            measurement = stereo_point_from_triplet(slam_db.manager_2d.link_triplet(frame_id, track_id))

            factor_index = graph.size()

            graph.add(
                gtsam.GenericStereoFactor3D(
                    measurement,
                    measurement_noise,
                    C(frame_id),
                    Q(track_id),
                    calibration,
                )
            )

            projection_factor_metadata.append(
                ProjectionFactorMetadata(
                    factor_index=factor_index,
                    frame_id=frame_id,
                    track_id=track_id,
                )
            )

    return graph, initial, inserted_landmarks, projection_factor_metadata
