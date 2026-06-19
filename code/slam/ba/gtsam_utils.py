import numpy as np
import gtsam

from slam.ba.results import BundleWindowSolution
from gtsam.symbol_shorthand import C, Q

def stereo_image_distances(
    measurement: gtsam.StereoPoint2,
    projection: gtsam.StereoPoint2,
) -> tuple[float, float]:
    """
    Return Euclidean reprojection distances in the left and right images.
    """
    left_distance = np.hypot(
        projection.uL() - measurement.uL(),
        projection.v() - measurement.v(),
    )
    right_distance = np.hypot(
        projection.uR() - measurement.uR(),
        projection.v() - measurement.v(),
    )
    return float(left_distance), float(right_distance)


def make_gtsam_stereo_calibration(P1: np.ndarray, P2: np.ndarray) -> gtsam.Cal3_S2Stereo:
    """
    Creates GTSAM stereo calibration from KITTI-style projection matrices.

    Assumes:
        P2[0, 3] = -fx * baseline
    """
    fx = P1[0, 0]
    fy = P1[1, 1]
    skew = P1[0, 1]
    cx = P1[0, 2]
    cy = P1[1, 2]

    baseline = -P2[0, 3] / fx

    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)


def convert_extrinsic_to_pose3(T_global_to_camera: np.ndarray) -> gtsam.Pose3:
    """
    Converts an extrinsic camera matrix to a GTSAM Pose3.

    Input convention:
        x_camera = R x_global + t

    GTSAM Pose3 convention:
        x_global = R_gtsam x_camera + t_gtsam

    Therefore:
        R_gtsam = R^T
        t_gtsam = -R^T t
    """
    if T_global_to_camera.shape == (4, 4) or T_global_to_camera.shape == (3, 4):
        R = T_global_to_camera[:3, :3]
        t = T_global_to_camera[:3, 3]
    else:
        raise ValueError(
            f"Expected transform of shape (3, 4) or (4, 4), "
            f"got {T_global_to_camera.shape}"
        )

    R_camera_to_global = R.T
    t_camera_to_global = -R.T @ t

    return gtsam.Pose3(
        gtsam.Rot3(R_camera_to_global),
        gtsam.Point3(t_camera_to_global),
    )


def make_stereo_camera(
    T_global_to_camera: np.ndarray,
    K: gtsam.Cal3_S2Stereo,
) -> gtsam.StereoCamera:
    """
    Creates a GTSAM StereoCamera from a global extrinsic matrix and stereo calibration.
    """
    pose = convert_extrinsic_to_pose3(T_global_to_camera)
    return gtsam.StereoCamera(pose, K)


def stereo_point_from_triplet(triplet: tuple[float, float, float]) -> gtsam.StereoPoint2:
    """
    Converts a TrackingDB triplet (x_left, x_right, y) into a GTSAM StereoPoint2.
    """
    x_left, x_right, y = triplet
    return gtsam.StereoPoint2(x_left, x_right, y)


def stereo_point_to_array(point: gtsam.StereoPoint2) -> np.ndarray:
    """
    Converts a GTSAM StereoPoint2 into a numpy array [u_left, u_right, v].
    """
    return np.array([point.uL(), point.uR(), point.v()], dtype=float)

def stereo_image_projection_distances(
    measurement: gtsam.StereoPoint2,
    projection: gtsam.StereoPoint2,
) -> tuple[float, float]:
    """
    Euclidean pixel distance between a measured and projected stereo point,
    separately in the left and right images.

    Left image uses (uL, v).
    Right image uses (uR, v).
    """
    left_distance = float(
        np.hypot(
            projection.uL() - measurement.uL(),
            projection.v() - measurement.v(),
        )
    )

    right_distance = float(
        np.hypot(
            projection.uR() - measurement.uR(),
            projection.v() - measurement.v(),
        )
    )

    return left_distance, right_distance


def compose_global_keyframe_poses(
    solutions: list[BundleWindowSolution],
) -> tuple[dict[int, gtsam.Pose3], dict[int, gtsam.Pose3]]:
    """
    Return:
      global_keyframe_poses[frame_id]:
          pose mapping that camera's coordinates into frame-0 coordinates.

      relative_keyframe_poses[end_frame]:
          pose mapping end-keyframe coordinates into predecessor-keyframe
          coordinates.
    """
    if not solutions:
        raise ValueError("No solved bundle windows.")

    first_keyframe = solutions[0].start_frame

    global_keyframe_poses = {
        first_keyframe: gtsam.Pose3(),
    }
    relative_keyframe_poses = {}

    for solution in solutions:
        start_frame = solution.start_frame
        end_frame = solution.end_frame
        result = solution.result

        start_pose_local = result.optimized.atPose3(C(start_frame))
        end_pose_local = result.optimized.atPose3(C(end_frame))

        # Maps coordinates in end_frame -> coordinates in start_frame.
        relative_pose = start_pose_local.between(end_pose_local)

        relative_keyframe_poses[end_frame] = relative_pose

        # Maps coordinates in end_frame -> coordinates in frame 0.
        global_keyframe_poses[end_frame] = (
            global_keyframe_poses[start_frame].compose(relative_pose)
        )

    return global_keyframe_poses, relative_keyframe_poses

def collect_global_landmarks(
    solutions: list[BundleWindowSolution],
    global_keyframe_poses: dict[int, gtsam.Pose3],
) -> np.ndarray:
    """
    Collect one global-frame landmark estimate per track.

    If a track appears in multiple independently optimized windows, retain its
    first estimate to avoid plotting several nearly-identical copies.
    """
    landmarks_by_track = {}

    for solution in solutions:
        result = solution.result
        local_to_global = global_keyframe_poses[solution.start_frame]

        for track_id in result.track_ids:
            if track_id in landmarks_by_track:
                continue

            try:
                point_local = result.optimized.atPoint3(Q(track_id))
            except RuntimeError:
                continue

            point_global = local_to_global.transformFrom(point_local)

            landmarks_by_track[track_id] = np.asarray(
                point_global,
                dtype=float,
            ).reshape(3)

    if not landmarks_by_track:
        return np.empty((0, 3), dtype=float)

    return np.vstack(list(landmarks_by_track.values()))
