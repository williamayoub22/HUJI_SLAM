"""Compose local bundle-adjustment window results into frame-0 coordinates."""

import gtsam
import numpy as np
from gtsam.symbol_shorthand import C, Q

from .results import BundleWindowSolution


def compose_keyframe_poses_in_frame0(
    solutions: list[BundleWindowSolution],
) -> tuple[dict[int, gtsam.Pose3], dict[int, gtsam.Pose3]]:
    """Compose consecutive local BA windows into the frame-0 coordinate system.

    Args:
        solutions: Solved local BA windows ordered by increasing frame index.

    Returns:
        A pair containing:

        - poses mapping each keyframe's coordinates into frame-0 coordinates;
        - relative poses mapping each window end frame into its start-frame
          coordinates.

    Raises:
        ValueError: If no solved windows are provided.
    """
    if not solutions:
        raise ValueError("No solved bundle windows.")

    first_keyframe = solutions[0].start_frame
    keyframe_poses_in_frame0 = {first_keyframe: gtsam.Pose3()}
    relative_keyframe_poses: dict[int, gtsam.Pose3] = {}

    for solution in solutions:
        start_frame = solution.start_frame
        end_frame = solution.end_frame

        start_pose_local = solution.result.optimized.atPose3(C(start_frame))
        end_pose_local = solution.result.optimized.atPose3(C(end_frame))

        # Maps end-frame coordinates into start-frame coordinates.
        relative_pose = start_pose_local.between(end_pose_local)
        relative_keyframe_poses[end_frame] = relative_pose

        # Maps end-frame coordinates into frame-0 coordinates.
        keyframe_poses_in_frame0[end_frame] = keyframe_poses_in_frame0[start_frame].compose(
            relative_pose
        )

    return keyframe_poses_in_frame0, relative_keyframe_poses


def collect_landmarks_in_frame0(
    solutions: list[BundleWindowSolution],
    keyframe_poses_in_frame0: dict[int, gtsam.Pose3],
) -> np.ndarray:
    """Collect one optimized landmark estimate per track in frame-0 coordinates.

    A track can occur in several independently optimized windows. The first
    encountered estimate is retained to avoid duplicate plotted landmarks.

    Args:
        solutions: Solved local BA windows.
        keyframe_poses_in_frame0: Pose of each window start frame in frame-0
            coordinates.

    Returns:
        Landmark coordinates with shape ``(num_landmarks, 3)``. Returns an
        empty ``(0, 3)`` array when no landmarks are available.
    """
    landmarks_by_track: dict[int, np.ndarray] = {}

    for solution in solutions:
        local_to_frame0 = keyframe_poses_in_frame0[solution.start_frame]

        for track_id in solution.result.track_ids:
            if track_id in landmarks_by_track:
                continue

            try:
                point_local = solution.result.optimized.atPoint3(Q(track_id))
            except RuntimeError:
                continue

            point_in_frame0 = local_to_frame0.transformFrom(point_local)

            landmarks_by_track[track_id] = np.asarray(
                point_in_frame0,
                dtype=float,
            ).reshape(3)

    if not landmarks_by_track:
        return np.empty((0, 3), dtype=float)

    return np.vstack(list(landmarks_by_track.values()))
