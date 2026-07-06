"""Construct local stereo bundle-adjustment factor graphs."""

import gtsam
import numpy as np
from gtsam.symbol_shorthand import C, Q

from slam.tracking_database import TrackingDB

from ..geometry.transforms import relative_world_to_camera_extrinsic
from .gtsam_utils import (
    make_stereo_camera,
    pose3_from_world_to_camera_extrinsic,
    stereo_point_from_triplet,
)
from .results import ProjectionFactorMetadata
from .window_selection import collect_window_tracks


def build_local_bundle_graph(
    db: TrackingDB,
    world_to_camera_extrinsics: np.ndarray,
    calibration: gtsam.Cal3_S2Stereo,
    window_frames: list[int],
    min_track_observations: int = 2,
    measurement_sigma_pixels: float = 1.0,
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
        db: Tracking database containing stereo observations.
        world_to_camera_extrinsics: World-to-camera extrinsics for all frames.
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

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    first_frame = window_frames[0]
    T_world_to_first = world_to_camera_extrinsics[first_frame]

    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        3,
        measurement_sigma_pixels,
    )

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-3, dtype=float))

    tracks_to_frames = collect_window_tracks(
        db=db,
        window_frames=window_frames,
        min_observations=min_track_observations,
    )

    # Construct candidate cameras before choosing which poses enter the graph.
    # Backprojection returns landmarks in the first-frame coordinate system.
    stereo_cameras: dict[int, gtsam.StereoCamera] = {}
    poses: dict[int, gtsam.Pose3] = {}
    for frame_id in window_frames:
        T_world_to_frame = world_to_camera_extrinsics[frame_id]

        T_first_to_frame = relative_world_to_camera_extrinsic(
            world_to_reference=T_world_to_first,
            world_to_frame=T_world_to_frame,
        )

        poses[frame_id] = pose3_from_world_to_camera_extrinsic(T_first_to_frame)

        stereo_cameras[frame_id] = make_stereo_camera(
            T_first_to_frame,
            calibration,
        )

    # Retain only tracks whose initialization measurement can be backprojected.
    initialized_landmarks: dict[int, gtsam.Point3] = {}
    valid_tracks_to_frames: dict[int, list[int]] = {}

    for track_id, track_frames in tracks_to_frames.items():
        initialization_frame = track_frames[-1]
        initialization_measurement = stereo_point_from_triplet(
            db.link_triplet(initialization_frame, track_id)
        )

        try:
            landmark = stereo_cameras[initialization_frame].backproject(initialization_measurement)
        except RuntimeError:
            continue

        initialized_landmarks[track_id] = landmark
        valid_tracks_to_frames[track_id] = track_frames

    if not valid_tracks_to_frames:
        raise RuntimeError("No valid landmarks could be initialized in this window.")

    active_frames = {first_frame}
    for track_frames in valid_tracks_to_frames.values():
        active_frames.update(track_frames)

    # Insert only poses that are anchored or referenced by projection factors.
    for frame_id in window_frames:
        if frame_id in active_frames:
            initial.insert(C(frame_id), poses[frame_id])

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
            measurement = stereo_point_from_triplet(db.link_triplet(frame_id, track_id))

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
