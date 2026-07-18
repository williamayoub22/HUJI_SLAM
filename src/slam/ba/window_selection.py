"""Select keyframes, local BA windows, and tracks for bundle adjustment."""

from collections.abc import Sequence

import numpy as np
from src.slam.database.tracking_database import TrackingDB

from src.slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic

# 1. Choose keyframes by motion:
#    - min gap: 5
#    - max gap: 18 or 20
#    - translation threshold: around 2–3 m
#    - rotation threshold: around 10–15 deg
#
# 2. Build local BA windows between adjacent keyframes.
#
# 3. Extract relative-pose constraints from BA.
#
# 4. Inflate covariance:
#    Sigma = alpha * Sigma_BA + Sigma_floor
#
# 5. For loop-closure candidate detection, use the covariance graph with these inflated edges.
#
# 6. Calibrate Mahalanobis threshold empirically as you already did.


def collect_window_tracks(
    db: TrackingDB,
    window_frames: Sequence[int],
    min_observations: int = 2,
) -> dict[int, list[int]]:
    """Return tracks observed sufficiently often within one frame window.

    Args:
        db: Tracking database containing track-to-frame associations.
        window_frames: Frame IDs included in the local BA window.
        min_observations: Minimum in-window observations required for a track.

    Returns:
        Mapping from each retained track ID to its sorted in-window frame IDs.

    Raises:
        ValueError: If ``min_observations`` is less than two.
    """
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2.")

    window_set = set(window_frames)
    tracks_to_frames: dict[int, list[int]] = {}

    for track_id in db.all_tracks():
        observed_frames = [frame_id for frame_id in db.frames(track_id) if frame_id in window_set]

        if len(observed_frames) >= min_observations:
            tracks_to_frames[track_id] = sorted(observed_frames)

    return tracks_to_frames


def choose_keyframes_by_interval(
    num_frames: int,
    step: int = 10,
) -> list[int]:
    """Choose keyframes at a fixed frame interval, including the final frame.

    Args:
        num_frames: Total number of frames in the sequence.
        step: Number of frames between consecutive regular keyframes.

    Returns:
        Ordered keyframe IDs beginning at zero and ending at ``num_frames - 1``.

    Raises:
        ValueError: If ``num_frames`` or ``step`` is not positive.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    if step <= 0:
        raise ValueError("step must be positive.")

    keyframes = list(range(0, num_frames, step))

    if keyframes[-1] != num_frames - 1:
        keyframes.append(num_frames - 1)

    return keyframes


def relative_motion_from_last_keyframe(last_kf_np_pose, current_np_pose) -> tuple[float, float]:
    last_pose = pose3_from_world_to_camera_extrinsic(last_kf_np_pose)
    current_pose = pose3_from_world_to_camera_extrinsic(current_np_pose)
    relative_pose = last_pose.between(current_pose)
    translation = float(np.linalg.norm(relative_pose.translation()))

    # Axis-angle rotation magnitude: trace(R) = 1 + 2 cos(theta).
    relative_rotation = relative_pose.rotation().matrix()
    cosine_angle = 0.5 * (np.trace(relative_rotation) - 1.0)
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    rotation_deg = float(np.degrees(np.arccos(cosine_angle)))

    return translation, rotation_deg


def choose_keyframes_by_motion(
    poses,
    min_gap: int = 11,
    max_gap: int = 20,
    min_translation: float = 5.0,
    min_rotation_deg: float = 12.0,
    target_translation: float = 5.0,
    target_rotation_deg: float = 12.0,
) -> list[int]:
    """Choose keyframes according to estimated camera motion.

    ``poses`` is the world-to-camera extrinsic array. Starting from the last
    selected keyframe, we inspect only the next bounded candidate range and pick
    the candidate whose motion is closest to the desired skeleton spacing. This
    avoids greedily accepting the first frame that passes the threshold.
    """
    keyframes = [0]
    last_kf = 0
    final_frame = len(poses) - 1

    while last_kf < final_frame:
        first_candidate = min(last_kf + min_gap, final_frame)
        last_candidate = min(last_kf + max_gap, final_frame)

        if first_candidate == final_frame:
            keyframes.append(final_frame)
            break

        candidates = []

        for frame_id in range(first_candidate, last_candidate + 1):
            translation, rotation_deg = relative_motion_from_last_keyframe(
                poses[last_kf],
                poses[frame_id],
            )

            enough_motion = translation >= min_translation or rotation_deg >= min_rotation_deg

            candidates.append(
                (
                    frame_id,
                    translation,
                    rotation_deg,
                    enough_motion,
                )
            )

        valid_candidates = [candidate for candidate in candidates if candidate[3]]

        if valid_candidates:
            chosen_frame, _, _, _ = min(
                valid_candidates,
                key=lambda candidate: (
                    abs(candidate[1] - target_translation) / target_translation
                    + abs(candidate[2] - target_rotation_deg) / target_rotation_deg
                ),
            )
        else:
            chosen_frame = last_candidate

        if chosen_frame == keyframes[-1]:
            break

        keyframes.append(chosen_frame)
        last_kf = chosen_frame

    if keyframes[-1] != final_frame:
        keyframes.append(final_frame)

    return keyframes


def bundle_windows_from_keyframes(
    keyframes: Sequence[int],
) -> list[list[int]]:
    """Return one inclusive frame window for every adjacent keyframe pair.

    Args:
        keyframes: Strictly increasing ordered keyframe IDs.

    Returns:
        Consecutive frame windows. Adjacent windows share one boundary keyframe.

    Raises:
        ValueError: If fewer than two keyframes are provided or keyframes are
            not strictly increasing.
    """
    if len(keyframes) < 2:
        raise ValueError("At least two keyframes are required.")

    if any(
        next_frame <= frame for frame, next_frame in zip(keyframes, keyframes[1:], strict=False)
    ):
        raise ValueError("Keyframes must be strictly increasing.")

    return [
        list(range(start_frame, end_frame + 1))
        for start_frame, end_frame in zip(
            keyframes[:-1],
            keyframes[1:],
            strict=False,
        )
    ]
