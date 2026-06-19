import numpy as np
import gtsam
from gtsam.symbol_shorthand import C, Q

from slam.tracking_database import TrackingDB
from .gtsam_utils import (
    convert_extrinsic_to_pose3,
    make_stereo_camera,
    stereo_point_from_triplet
)
from .window_selection import collect_window_tracks
from .results import ProjectionFactorMetadata
from ..geometry.transforms import relative_extrinsic


def build_local_bundle_graph(
    db: TrackingDB,
    global_camera_matrices: np.ndarray,
    K: gtsam.Cal3_S2Stereo,
    window_frames: list[int],
    min_track_observations: int = 2,
    measurement_sigma_pixels: float = 1.0,
) -> tuple[
    gtsam.NonlinearFactorGraph,
    gtsam.Values,
    list[int],
    list[ProjectionFactorMetadata],
]:
    """
    Builds a local bundle adjustment graph for a consecutive frame window.

    All poses and landmarks are represented in the coordinate system of the
    first frame in the window.

    Variables:
        C(frame_id): camera pose for frame_id
        Q(track_id): 3D landmark for track_id

    Factors:
        - Prior on the first camera pose to remove gauge freedom.
        - Stereo projection factors between observed cameras and landmarks.
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        3,
        measurement_sigma_pixels,
    )

    first_frame = window_frames[0]
    T_global_to_first = global_camera_matrices[first_frame]

    # Select tracks before inserting poses. GTSAM requires every value in the
    # initial estimate to be referenced by at least one graph factor.
    tracks_to_frames = collect_window_tracks(
        db=db,
        window_frames=window_frames,
        min_observations=min_track_observations,
    )

    active_frames = {
        frame_id
        for track_frames in tracks_to_frames.values()
        for frame_id in track_frames
    }
    active_frames.add(first_frame)  # Needed by the anchoring prior.


    stereo_cameras = {}

    # Insert camera poses in first-frame coordinates.
    for frame_id in window_frames:
        if frame_id not in active_frames:
            continue

        T_global_to_frame = global_camera_matrices[frame_id]

        T_first_to_frame = relative_extrinsic(
            T_global_to_ref=T_global_to_first,
            T_global_to_frame=T_global_to_frame,
        )

        pose = convert_extrinsic_to_pose3(T_first_to_frame)
        camera = make_stereo_camera(T_first_to_frame, K)

        initial.insert(C(frame_id), pose)
        stereo_cameras[frame_id] = camera

    # Anchor first pose to avoid gauge freedom.
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    )

    graph.add(
        gtsam.PriorFactorPose3(
            C(first_frame),
            initial.atPose3(C(first_frame)),
            prior_noise,
        )
    )

    # tracks_to_frames = collect_window_tracks(
    #     db=db,
    #     window_frames=window_frames,
    #     min_observations=min_track_observations,
    # )

    inserted_landmarks: list[int] = []
    projection_factor_metadata: list[ProjectionFactorMetadata] = []

    for track_id, track_frames in tracks_to_frames.items():
        init_frame = track_frames[-1]

        init_measurement = stereo_point_from_triplet(
            db.link_triplet(init_frame, track_id)
        )

        try:
            landmark = stereo_cameras[init_frame].backproject(init_measurement)
        except RuntimeError:
            continue

        initial.insert(Q(track_id), landmark)
        inserted_landmarks.append(track_id)

        for frame_id in track_frames:
            measurement = stereo_point_from_triplet(
                db.link_triplet(frame_id, track_id)
            )

            factor_index = graph.size()
            factor = gtsam.GenericStereoFactor3D(
                measurement,
                measurement_noise,
                C(frame_id),
                Q(track_id),
                K,
            )
            graph.add(factor)
            projection_factor_metadata.append(
                ProjectionFactorMetadata(
                    factor_index=factor_index,
                    frame_id=frame_id,
                    track_id=track_id,
                )
            )

    return graph, initial, inserted_landmarks, projection_factor_metadata
