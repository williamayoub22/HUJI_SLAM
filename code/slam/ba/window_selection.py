import numpy as np

# from graph_builder import build_local_bundle_graph
from gtsam.symbol_shorthand import C, Q

from slam.tracking_database import TrackingDB


def _bundle_camera_centers(result) -> np.ndarray:
    return np.vstack([
        np.asarray(
            result.optimized.atPose3(C(frame_id)).translation(),
            dtype=float,
        ).reshape(3)
        for frame_id in result.window_frames
    ])


def _bundle_landmarks(result) -> np.ndarray:
    """Extracts optimized landmark locations for a local bundle window."""
    landmarks = []

    for track_id in result.track_ids:
        try:
            point = result.optimized.atPoint3(Q(track_id))
        except RuntimeError:
            continue

        landmarks.append(np.asarray(point, dtype=float).reshape(3))

    return (
        np.vstack(landmarks)
        if landmarks
        else np.empty((0, 3), dtype=float)
    )


def collect_window_tracks(
    db: TrackingDB,
    window_frames: list[int],
    min_observations: int = 2,
) -> dict[int, list[int]]:
    """
    Returns tracks observed in the given window.

    Output:
        track_id -> sorted list of frame_ids where the track is observed.
    """
    window_set = set(window_frames)
    tracks_to_frames: dict[int, list[int]] = {}

    for track_id in db.all_tracks():
        frames = [
            frame_id
            for frame_id in db.frames(track_id)
            if frame_id in window_set
        ]

        if len(frames) >= min_observations:
            tracks_to_frames[track_id] = sorted(frames)

    return tracks_to_frames


def choose_keyframes_by_interval(num_frames: int, step: int = 10) -> list[int]:
    """
    Chooses keyframes at a fixed frame interval.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    keyframes = list(range(0, num_frames, step))

    if keyframes[-1] != num_frames - 1:
        keyframes.append(num_frames - 1)

    return keyframes


def first_bundle_window_from_keyframes(
    keyframes: list[int],
) -> list[int]:
    """
    Returns all consecutive frames between the first two keyframes.
    """
    first = keyframes[0]
    second = keyframes[1]
    return list(range(first, second + 1))

def bundle_windows_from_keyframes(
    keyframes: list[int],
) -> list[list[int]]:
    """
    Return one inclusive frame window for every adjacent keyframe pair.

    Example:
        [0, 10, 20] -> [[0, ..., 10], [10, ..., 20]]
    """
    if len(keyframes) < 2:
        raise ValueError("Need at least two keyframes.")

    return [
        list(range(start_frame, end_frame + 1))
        for start_frame, end_frame in zip(keyframes[:-1], keyframes[1:])
    ]

