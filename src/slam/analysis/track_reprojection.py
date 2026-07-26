"""Analyze stereo reprojection consistency along one tracked feature."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import C, Q
from src.slam.ba.gtsam_utils import (
    pose3_from_world_to_camera_extrinsic,
    stereo_point_from_triplet,
    stereo_residual_norm,
)
from src.slam.tracking_database import TrackingDB

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
    min_length: int = 10,
    seed: int = 0,
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
    if min_length < 1:
        raise ValueError("min_length must be positive.")

    candidate_tracks = db.tracks_with_min_length(min_length)

    if not candidate_tracks:
        raise RuntimeError(f"No tracks with length >= {min_length} were found.")

    rng = np.random.default_rng(seed)
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
    min_track_length: int = 10,
    seed: int = 0,
    sigma_pixels: float = 1.0,
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
        min_length=min_track_length,
        seed=seed,
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
        sigma_pixels=sigma_pixels,
    )

    return TrackReprojectionResult(
        track_id=track_id,
        frame_ids=frame_ids,
        landmark_world=landmark_world,
        reprojection_errors=reprojection_errors,
        factor_errors=factor_errors,
        covariance=covariance,
    )
