"""Select keyframes, local BA windows, and tracks for bundle adjustment."""

from collections.abc import Sequence

from slam.tracking_database import TrackingDB


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
