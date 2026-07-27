"""Analyze stereo reprojection consistency along one tracked feature."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import C, Q

from src.slam import config
from src.slam.config import GLOBAL_CAMERA_MATRICES_PATH
from src.slam.data.tracking_db import TrackingDB
from src.slam.geometry.stereo import (
    make_gtsam_stereo_calibration,
    make_stereo_camera,
    pose3_from_world_to_camera_extrinsic,
    stereo_image_distances,
    stereo_point_from_triplet,
    stereo_residual_norm,
)
from src.slam.geometry.transforms import compose_camera_transform, to_homogeneous_transform
from src.slam.io.calibration import read_stereo_calibration

LOGGER = logging.getLogger(__name__)


@dataclass
class TrackReprojectionResult:
    """Results of reprojection and factor-error analysis for one feature track.

    Attributes:
        track_id: Selected tracking-database track identifier.
        frame_ids: Ordered frames observing the track.
        landmark_world: Landmark initialized from the final track observation.
        reprojection_errors: Full stereo residual norm for each frame.
        factor_errors: GTSAM factor error for each frame.
        covariance: Stereo measurement covariance used by the factors.
    """

    track_id: int
    frame_ids: list[int]
    landmark_world: gtsam.Point3
    reprojection_errors: np.ndarray
    factor_errors: np.ndarray
    covariance: np.ndarray


def choose_track_with_min_length(
    db: TrackingDB,
) -> int:
    """Select a reproducible random track with at least ``min_length`` frames.

    Args:
        db: Tracking database from which to choose a track.
        min_length: Minimum required number of observations.
        seed: Random seed used for reproducible selection.

    Returns:
        Selected track ID.

    Raises:
        ValueError: If ``min_length`` is less than one.
        RuntimeError: If no track satisfies the length requirement.
    """
    if config.MIN_TRACK_LENGTH < 1:
        raise ValueError("min_length must be positive.")

    candidate_tracks = db.tracks_with_min_length(config.MIN_TRACK_LENGTH)

    if not candidate_tracks:
        raise RuntimeError(f"No tracks with length >= {config.MIN_TRACK_LENGTH} were found.")

    rng = np.random.default_rng(config.RANDOM_SEED)
    return int(rng.choice(candidate_tracks))


def _collect_track_measurements(
    db: TrackingDB,
    track_id: int,
    frame_ids: Sequence[int],
) -> dict[int, gtsam.StereoPoint2]:
    """Convert all observations of one track to GTSAM stereo measurements."""
    return {
        frame_id: stereo_point_from_triplet(db.link_triplet(frame_id, track_id))
        for frame_id in frame_ids
    }


def _build_track_cameras_and_poses(
    frame_ids: Sequence[int],
    world_to_camera_extrinsics: np.ndarray,
    calibration: gtsam.Cal3_S2Stereo,
) -> tuple[dict[int, gtsam.StereoCamera], dict[int, gtsam.Pose3]]:
    """Build GTSAM cameras and camera-to-world poses for the selected frames."""
    cameras: dict[int, gtsam.StereoCamera] = {}
    poses: dict[int, gtsam.Pose3] = {}

    for frame_id in frame_ids:
        world_to_camera = world_to_camera_extrinsics[frame_id]

        pose = pose3_from_world_to_camera_extrinsic(world_to_camera)

        poses[frame_id] = pose
        cameras[frame_id] = gtsam.StereoCamera(pose, calibration)

    return cameras, poses


def _triangulate_from_final_observation(
    frame_ids: Sequence[int],
    cameras: dict[int, gtsam.StereoCamera],
    measurements: dict[int, gtsam.StereoPoint2],
) -> gtsam.Point3:
    """Backproject the final observation of a track into world coordinates."""
    final_frame_id = frame_ids[-1]
    return cameras[final_frame_id].backproject(measurements[final_frame_id])


def _compute_reprojection_errors(
    frame_ids: Sequence[int],
    cameras: dict[int, gtsam.StereoCamera],
    measurements: dict[int, gtsam.StereoPoint2],
    landmark_world: gtsam.Point3,
) -> np.ndarray:
    """Compute full stereo reprojection residual norms for all track frames."""
    errors: list[float] = []

    for frame_id in frame_ids:
        try:
            projection = cameras[frame_id].project(landmark_world)
            error = stereo_residual_norm(measurements[frame_id], projection)
        except RuntimeError as error:
            LOGGER.warning(
                "Could not project track landmark into frame %d: %s",
                frame_id,
                error,
            )
            error = np.nan

        errors.append(error)

    return np.asarray(errors, dtype=float)


def _compute_factor_errors(
    frame_ids: Sequence[int],
    poses: dict[int, gtsam.Pose3],
    measurements: dict[int, gtsam.StereoPoint2],
    landmark_world: gtsam.Point3,
    track_id: int,
    calibration: gtsam.Cal3_S2Stereo,
    sigma_pixels: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one stereo projection factor per observation of the track."""
    if sigma_pixels <= 0:
        raise ValueError("sigma_pixels must be positive.")

    covariance = sigma_pixels**2 * np.eye(3)
    noise_model = gtsam.noiseModel.Isotropic.Sigma(3, sigma_pixels)

    values = gtsam.Values()
    landmark_key = Q(track_id)
    values.insert(landmark_key, landmark_world)

    for frame_id in frame_ids:
        values.insert(C(frame_id), poses[frame_id])

    errors: list[float] = []

    for frame_id in frame_ids:
        factor = gtsam.GenericStereoFactor3D(
            measurements[frame_id],
            noise_model,
            C(frame_id),
            landmark_key,
            calibration,
        )

        try:
            error = float(factor.error(values))
        except RuntimeError as exception:
            LOGGER.warning(
                "Could not evaluate track factor for frame %d: %s",
                frame_id,
                exception,
            )
            error = np.nan

        errors.append(error)

    return np.asarray(errors, dtype=float), covariance


def analyze_track_reprojection(
    db: TrackingDB,
    world_to_camera_extrinsics: np.ndarray,
    calibration: gtsam.Cal3_S2Stereo,
) -> TrackReprojectionResult:
    """Analyze reprojection and factor errors for one reproducibly selected track.

    The landmark is initialized by backprojecting the last stereo observation.
    It is then projected into every frame that observes the same track.

    Args:
        db: Tracking database containing stereo feature observations.
        world_to_camera_extrinsics: World-to-camera matrices for all frames.
        calibration: Stereo camera calibration.
        min_track_length: Minimum length required for the selected track.
        seed: Random seed for reproducible track selection.
        sigma_pixels: Isotropic standard deviation of each stereo measurement.

    Returns:
        Reprojection residuals, factor errors, and measurement covariance.
    """
    track_id = choose_track_with_min_length(
        db=db,
    )

    frame_ids = list(db.frames(track_id))
    measurements = _collect_track_measurements(db, track_id, frame_ids)

    cameras, poses = _build_track_cameras_and_poses(
        frame_ids=frame_ids,
        world_to_camera_extrinsics=world_to_camera_extrinsics,
        calibration=calibration,
    )

    landmark_world = _triangulate_from_final_observation(
        frame_ids=frame_ids,
        cameras=cameras,
        measurements=measurements,
    )

    reprojection_errors = _compute_reprojection_errors(
        frame_ids=frame_ids,
        cameras=cameras,
        measurements=measurements,
        landmark_world=landmark_world,
    )

    factor_errors, covariance = _compute_factor_errors(
        frame_ids=frame_ids,
        poses=poses,
        measurements=measurements,
        landmark_world=landmark_world,
        track_id=track_id,
        calibration=calibration,
        sigma_pixels=config.MEASUREMENT_SIGMA_PIXELS,
    )

    return TrackReprojectionResult(
        track_id=track_id,
        frame_ids=frame_ids,
        landmark_world=landmark_world,
        reprojection_errors=reprojection_errors,
        factor_errors=factor_errors,
        covariance=covariance,
    )



def compute_median_projection_errors_by_distance(database, world_to_camera_by_frame, calibration, *, optimize_landmarks, min_track_length=2, max_distance=50):
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
    local_stereo_camera = make_stereo_camera(np.eye(4, dtype=float), calibration)
    for track_id in manager_2d.all_tracks():
        track_frames = sorted(manager_2d.frames(track_id))
        usable_frames = [frame_id for frame_id in track_frames if frame_id in world_to_camera_by_frame]
        if len(usable_frames) < min_track_length:
            continue
        triangulation_frame = usable_frames[-1]
        evaluation_frames = [frame_id for frame_id in usable_frames if 0 <= triangulation_frame - frame_id <= max_distance]
        if len(evaluation_frames) < min_track_length:
            continue
        try:
            if optimize_landmarks:
                landmark_world = optimize_track_landmark(manager_2d=manager_2d, track_id=track_id, frame_ids=evaluation_frames, world_to_camera_by_frame=world_to_camera_by_frame, calibration=calibration)
                landmark_world = np.asarray(landmark_world, dtype=float).reshape(3)
                if not np.all(np.isfinite(landmark_world)):
                    raise ValueError('Optimized landmark contains non-finite values.')
            else:
                triangulation_measurement = stereo_point_from_triplet(get_stereo_observation(manager_2d, triangulation_frame, track_id))
                landmark_in_triangulation_camera = np.asarray(local_stereo_camera.backproject(triangulation_measurement), dtype=float).reshape(3)
                if not np.all(np.isfinite(landmark_in_triangulation_camera)):
                    raise ValueError('Triangulated landmark contains non-finite values.')
        except (KeyError, IndexError, ValueError, RuntimeError, Exception):
            rejected_tracks += 1
            continue
        track_contributed = False
        for frame_id in evaluation_frames:
            distance = triangulation_frame - frame_id
            try:
                if optimize_landmarks:
                    frame_extrinsic = to_homogeneous_transform(world_to_camera_by_frame[frame_id])
                    frame_camera = make_stereo_camera(frame_extrinsic, calibration)
                    projection = frame_camera.project(gtsam.Point3(landmark_world))
                else:
                    frame_from_triangulation = compose_camera_transform(world_to_camera_by_frame=world_to_camera_by_frame, source_frame=triangulation_frame, target_frame=frame_id)
                    landmark_homogeneous = np.append(landmark_in_triangulation_camera, 1.0)
                    landmark_in_frame_camera = (frame_from_triangulation @ landmark_homogeneous)[:3]
                    if not np.all(np.isfinite(landmark_in_frame_camera)):
                        continue
                    projection = local_stereo_camera.project(gtsam.Point3(landmark_in_frame_camera))
                measurement = stereo_point_from_triplet(get_stereo_observation(manager_2d, frame_id, track_id))
                left_error, right_error = stereo_image_distances(measurement=measurement, projection=projection)
            except (KeyError, IndexError, ValueError, RuntimeError, Exception):
                continue
            left_errors_by_distance[distance].append(float(left_error))
            right_errors_by_distance[distance].append(float(right_error))
            track_contributed = True
        if track_contributed:
            used_tracks += 1
        else:
            rejected_tracks += 1
    distances = np.asarray(sorted(set(left_errors_by_distance) & set(right_errors_by_distance)), dtype=int)
    median_left_errors = np.asarray([np.median(left_errors_by_distance[distance]) for distance in distances], dtype=float)
    median_right_errors = np.asarray([np.median(right_errors_by_distance[distance]) for distance in distances], dtype=float)
    sample_counts = np.asarray([min(len(left_errors_by_distance[distance]), len(right_errors_by_distance[distance])) for distance in distances], dtype=int)
    print(f'Used tracks:        {used_tracks}')
    print(f'Rejected tracks:    {rejected_tracks}')
    print(f'Computed distances: {len(distances)}')
    return (distances, median_left_errors, median_right_errors, sample_counts)

def optimize_track_landmark(manager_2d, track_id, frame_ids, world_to_camera_by_frame, calibration, measurement_sigma_pixels=1.0):
    """Optimize one landmark using all stereo observations in one track.

    Camera poses are fixed using tight Pose3 priors. Only the landmark is
    meaningfully adjusted by the optimization.
    """
    if len(frame_ids) < 2:
        raise ValueError('At least two observations are required.')
    triangulation_frame = frame_ids[-1]
    triangulation_extrinsic = to_homogeneous_transform(world_to_camera_by_frame[triangulation_frame])
    triangulation_camera = make_stereo_camera(triangulation_extrinsic, calibration)
    triangulation_measurement = stereo_point_from_triplet(get_stereo_observation(manager_2d, triangulation_frame, track_id))
    initial_landmark = triangulation_camera.backproject(triangulation_measurement)
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    landmark_key = Q(int(track_id))
    initial_values.insert(landmark_key, gtsam.Point3(np.asarray(initial_landmark, dtype=float)))
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, measurement_sigma_pixels)
    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-07, dtype=float))
    for frame_id in frame_ids:
        pose_key = C(int(frame_id))
        world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[frame_id])
        camera_pose = pose3_from_world_to_camera_extrinsic(world_to_camera)
        initial_values.insert(pose_key, camera_pose)
        graph.add(gtsam.PriorFactorPose3(pose_key, camera_pose, pose_prior_noise))
        measurement = stereo_point_from_triplet(get_stereo_observation(manager_2d, frame_id, track_id))
        graph.add(gtsam.GenericStereoFactor3D(measurement, measurement_noise, pose_key, landmark_key, calibration))
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    optimized_values = optimizer.optimize()
    optimized_landmark = np.asarray(optimized_values.atPoint3(landmark_key), dtype=float).reshape(3)
    return optimized_landmark

def get_stereo_observation(manager_2d, frame_id, track_id):
    """Return the observation as (x_left, x_right, y).

    This isolates the database-specific API in one place. If manager_2d does
    not expose link_triplet directly, only this function needs to be changed.
    """
    if hasattr(manager_2d, 'link_triplet'):
        return manager_2d.link_triplet(frame_id, track_id)
    if hasattr(manager_2d, 'database') and hasattr(manager_2d.database, 'link_triplet'):
        return manager_2d.database.link_triplet(frame_id, track_id)
    raise AttributeError('Could not retrieve a stereo observation. Implement get_stereo_observation() using the manager_2d link API. It must return (x_left, x_right, y).')

def compute_pnp_analysis(database):
    """
    Compute temporal PnP reprojection errors using only estimated camera poses.

    A point is triangulated in the latest camera of each track and propagated
    into earlier cameras through the composed estimated camera motions.
    """
    print('\n' + '=' * 60)
    print('PnP temporal projection-error analysis')
    print('=' * 60)
    pnp_extrinsics = np.asarray(np.load(GLOBAL_CAMERA_MATRICES_PATH), dtype=float)
    estimated_world_to_camera_by_frame = {frame_id: to_homogeneous_transform(extrinsic) for frame_id, extrinsic in enumerate(pnp_extrinsics)}
    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)
    distances, median_left_errors, median_right_errors, sample_counts = compute_median_projection_errors_by_distance(database=database, world_to_camera_by_frame=estimated_world_to_camera_by_frame, calibration=calibration, optimize_landmarks=False, min_track_length=2, max_distance=50)
    return {'distances': distances, 'median_left_errors': median_left_errors, 'median_right_errors': median_right_errors, 'sample_counts': sample_counts}

def compute_bundle_adjustment_analysis(database):
    """Optimize each track landmark and compute BA reprojection errors."""
    print('\n' + '=' * 60)
    print('Track Bundle-Adjustment projection-error analysis')
    print('=' * 60)
    pnp_extrinsics = np.asarray(np.load(GLOBAL_CAMERA_MATRICES_PATH), dtype=float)
    world_to_camera_by_frame = {frame_id: to_homogeneous_transform(extrinsic) for frame_id, extrinsic in enumerate(pnp_extrinsics)}
    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)
    distances, median_left_errors, median_right_errors, sample_counts = compute_median_projection_errors_by_distance(database=database, world_to_camera_by_frame=world_to_camera_by_frame, calibration=calibration, optimize_landmarks=True, min_track_length=2, max_distance=50)
    return {'distances': distances, 'median_left_errors': median_left_errors, 'median_right_errors': median_right_errors, 'sample_counts': sample_counts}