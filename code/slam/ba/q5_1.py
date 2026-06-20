from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import gtsam
from gtsam.symbol_shorthand import C, Q

from slam.tracking_database import TrackingDB

from .results import TrackReprojectionResult
from ..ba.gtsam_utils import (
    convert_extrinsic_to_pose3,
    make_stereo_camera,
    stereo_point_from_triplet,
    stereo_point_to_array,
)


def choose_track_with_min_length(
    db: TrackingDB,
    min_length: int = 10,
    seed: int = 0,
) -> int:
    """
    Selects a reproducible arbitrary track with length at least min_length.
    """
    candidate_tracks = db.tracks_with_min_length(min_length)

    if len(candidate_tracks) == 0:
        raise RuntimeError(f"No tracks with length >= {min_length} were found.")

    rng = np.random.default_rng(seed)
    return int(rng.choice(candidate_tracks))


def collect_stereo_measurements(
    db: TrackingDB,
    track_id: int,
    frame_ids: Sequence[int],
) -> Dict[int, gtsam.StereoPoint2]:
    """
    Collects the stereo measurements of a track in GTSAM format.

    Each TrackingDB measurement is stored as:
        (x_left, x_right, y)

    GTSAM expects:
        StereoPoint2(x_left, x_right, y)
    """
    return {
        frame_id: stereo_point_from_triplet(db.link_triplet(frame_id, track_id))
        for frame_id in frame_ids
    }


def build_track_cameras_and_poses(
    frame_ids: Sequence[int],
    global_camera_matrices: np.ndarray,
    K: gtsam.Cal3_S2Stereo,
) -> tuple[Dict[int, gtsam.StereoCamera], Dict[int, gtsam.Pose3]]:
    """
    Builds a GTSAM StereoCamera and Pose3 for every frame in the track.

    global_camera_matrices[frame_id] is assumed to be the extrinsic matrix:
        x_camera = R x_global + t

    GTSAM expects the opposite direction, so convert_extrinsic_to_pose3()
    is used internally.
    """
    cameras = {}
    poses = {}

    for frame_id in frame_ids:
        T_global_to_camera = global_camera_matrices[frame_id]

        pose = convert_extrinsic_to_pose3(T_global_to_camera)
        camera = make_stereo_camera(T_global_to_camera, K)

        poses[frame_id] = pose
        cameras[frame_id] = camera

    return cameras, poses


def triangulate_landmark_from_last_frame(
    frame_ids: Sequence[int],
    cameras: Dict[int, gtsam.StereoCamera],
    measurements: Dict[int, gtsam.StereoPoint2],
) -> gtsam.Point3:
    """
    Triangulates a 3D landmark in global coordinates using the last frame
    of the track.

    Since cameras[last_frame] is a GTSAM StereoCamera with a camera-to-global
    pose, backproject() returns the point in the global coordinate system.
    """
    last_frame_id = frame_ids[-1]
    last_measurement = measurements[last_frame_id]
    print("[5.1] Last-frame measurement:")
    print(f"  uL = {last_measurement.uL():.3f}")
    print(f"  uR = {last_measurement.uR():.3f}")
    print(f"  v  = {last_measurement.v():.3f}")
    print(f"  disparity uL-uR = {last_measurement.uL() - last_measurement.uR():.3f}")
    return cameras[last_frame_id].backproject(measurements[last_frame_id])


def compute_reprojection_errors(
    frame_ids: Sequence[int],
    cameras: Dict[int, gtsam.StereoCamera],
    measurements: Dict[int, gtsam.StereoPoint2],
    landmark_global: gtsam.Point3,
) -> np.ndarray:
    """
    Projects a global 3D point into all frames and computes stereo L2 errors.

    If the point is behind a camera, GTSAM raises a Stereo Cheirality Exception.
    In that case we store np.nan for that frame and print a warning.
    """
    errors = []

    for frame_id in frame_ids:
        try:
            predicted = cameras[frame_id].project(landmark_global)
            measured = measurements[frame_id]

            residual = stereo_point_to_array(measured) - stereo_point_to_array(predicted)
            errors.append(np.linalg.norm(residual))

        except RuntimeError as err:
            print(
                f"[5.1] WARNING: Could not project landmark into frame {frame_id}. "
                f"Reason: {err}"
            )
            errors.append(np.nan)

    return np.asarray(errors)


def compute_factor_errors(
    frame_ids: Sequence[int],
    poses: Dict[int, gtsam.Pose3],
    measurements: Dict[int, gtsam.StereoPoint2],
    landmark_global: gtsam.Point3,
    track_id: int,
    K: gtsam.Cal3_S2Stereo,
    sigma_pixels: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates one GenericStereoFactor3D per frame and evaluates its factor error.

    We use isotropic pixel noise:
        Sigma = sigma_pixels^2 * I_3

    The factor error is:
        0.5 * r^T Sigma^{-1} r

    Frames where the landmark is behind the camera are assigned np.nan.
    """
    noise_model = gtsam.noiseModel.Isotropic.Sigma(dim=3, sigma=sigma_pixels)
    covariance = (sigma_pixels ** 2) * np.eye(3)

    values = gtsam.Values()
    landmark_key = Q(track_id)
    values.insert(landmark_key, landmark_global)

    for frame_id in frame_ids:
        values.insert(C(frame_id), poses[frame_id])

    factor_errors = []

    for frame_id in frame_ids:
        factor = gtsam.GenericStereoFactor3D(
            measurements[frame_id],
            noise_model,
            C(frame_id),
            landmark_key,
            K,
        )

        try:
            factor_errors.append(factor.error(values))
        except RuntimeError as err:
            print(
                f"[5.1] WARNING: Could not evaluate factor error for frame {frame_id}. "
                f"Reason: {err}"
            )
            factor_errors.append(np.nan)

    return np.asarray(factor_errors), covariance


def run_track_reprojection_analysis(
    db: TrackingDB,
    global_camera_matrices: np.ndarray,
    K: gtsam.Cal3_S2Stereo,
    min_track_length: int = 10,
    seed: int = 0,
    sigma_pixels: float = 1.0,
) -> TrackReprojectionResult:
    """
    Runs the full Exercise 5.1 analysis for one arbitrary long track.
    """
    track_id = choose_track_with_min_length(
        db=db,
        min_length=min_track_length,
        seed=seed,
    )

    frame_ids = db.frames(track_id)

    measurements = collect_stereo_measurements(
        db=db,
        track_id=track_id,
        frame_ids=frame_ids,
    )

    cameras, poses = build_track_cameras_and_poses(
        frame_ids=frame_ids,
        global_camera_matrices=global_camera_matrices,
        K=K,
    )

    landmark_global = triangulate_landmark_from_last_frame(
        frame_ids=frame_ids,
        cameras=cameras,
        measurements=measurements,
    )

    reprojection_errors = compute_reprojection_errors(
        frame_ids=frame_ids,
        cameras=cameras,
        measurements=measurements,
        landmark_global=landmark_global,
    )

    factor_errors, covariance = compute_factor_errors(
        frame_ids=frame_ids,
        poses=poses,
        measurements=measurements,
        landmark_global=landmark_global,
        track_id=track_id,
        K=K,
        sigma_pixels=sigma_pixels,
    )

    return TrackReprojectionResult(
        track_id=track_id,
        frame_ids=list(frame_ids),
        landmark_global=landmark_global,
        reprojection_errors=reprojection_errors,
        factor_errors=factor_errors,
        covariance=covariance,
    )
