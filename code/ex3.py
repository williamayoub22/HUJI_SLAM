from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam.features.detectors import DEFAULT_ORB_NUM_FEATURES, FeatureType
from slam.geometry.correspondences import find_common_points
from slam.geometry.pnp import solve_pnp_safe
from slam.geometry.projection import count_supporters
from slam.geometry.ransac import ransac_pnp
from slam.geometry.transforms import (
    camera_centers_from_world_to_camera_extrinsics,
    compose_frame0_to_camera_extrinsics,
)
from slam.io.calibration import read_stereo_calibration
from slam.io.poses import read_ground_truth_poses
from slam.pipeline.stereo_pipeline import StereoPointCloud, create_stereo_point_cloud
from slam.pipeline.temporal_pipeline import match_left_frames
from slam.pipeline.tracking_pipeline import track_sequence
from slam.visualization.ex3_plots import (
    plot_four_image_matches,
    plot_task_3_3,
    plot_task_3_4,
    plot_task_3_5_matches,
    plot_task_3_5_point_clouds,
)
from slam.visualization.trajectory import plot_trajectory
from slam.visualization.visualization import plot_point_cloud_on_axis

FRAME_0_INDEX = 0
FRAME_1_INDEX = 1

COMMON_POINT_TOLERANCE = 1e-3
SUPPORTER_THRESHOLD_PIXELS = 2.0
PNP_NUM_POINTS = 4
FEATURE_TYPE: FeatureType = "akaze"
SEQ_DIR = Path(__file__).resolve().parent.parent / "dataset" / "sequences" / "00"
POSES_PATH = Path(__file__).resolve().parent.parent / "dataset" / "poses" / "00.txt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "ex3"


def section_3_1() -> tuple[StereoPointCloud, StereoPointCloud]:
    """
    Section 3.1:
    Creates point clouds for stereo pair 0 and stereo pair 1.

    Pair 0 is the point cloud from Exercise 2.
    Pair 1 is created from:
        left1:  VAN_ex/dataset/sequences/00/image_0/000001.png
        right1: VAN_ex/dataset/sequences/00/image_1/000001.png
    """
    print("--- Section 3.1 ---")

    point_cloud_0 = create_stereo_point_cloud(
        FRAME_0_INDEX,
        reject_negative_depth=True,
        feature_type="akaze",
        num_features=None,
        use_ratio_test=True,
    )

    point_cloud_1 = create_stereo_point_cloud(
        FRAME_1_INDEX,
        reject_negative_depth=True,
        feature_type="akaze",
        num_features=None,
        use_ratio_test=True,
    )

    fig = plt.figure(figsize=(16, 7))

    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_point_cloud_on_axis(
        ax0,
        point_cloud_0.points_3d,
        "Point Cloud from Stereo Pair 0",
        color="tab:blue",
    )

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_point_cloud_on_axis(
        ax1,
        point_cloud_1.points_3d,
        "Point Cloud from Stereo Pair 1",
        color="tab:orange",
    )

    plt.tight_layout()
    plt.show()

    return point_cloud_0, point_cloud_1


def section_3_2(
    point_cloud_0: StereoPointCloud,
    point_cloud_1: StereoPointCloud,
) -> tuple[np.ndarray, np.ndarray, list[cv2.DMatch]]:
    """
    Section 3.2:
    Matches features between the two left images: left_0 and left_1.
    """
    print("--- Section 3.2 ---")

    temporal_data = match_left_frames(
        FRAME_0_INDEX,
        FRAME_1_INDEX,
        feature_type=FEATURE_TYPE,
        num_features=DEFAULT_ORB_NUM_FEATURES,
        use_ratio_test=True,
    )

    print(
        f"Number of ratio-test matches between "
        f"left_{FRAME_0_INDEX} and left_{FRAME_1_INDEX}: {len(temporal_data.matches)}"
    )

    plot_four_image_matches(point_cloud_0, point_cloud_1, temporal_data)

    return temporal_data.left0_pts, temporal_data.left1_pts, temporal_data.matches


def section_3_3(
    point_cloud_0: StereoPointCloud,
    point_cloud_1: StereoPointCloud,
    left0_pts: np.ndarray,
    left1_pts: np.ndarray,
) -> np.ndarray:
    print("--- Section 3.3 ---")

    P1, P2 = read_stereo_calibration()
    K = P1[:, :3]

    correspondences = find_common_points(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
    )

    points_3d = correspondences.points_3d
    left0 = correspondences.left0
    right0 = correspondences.right0
    left1 = correspondences.left1
    right1 = correspondences.right1

    print(f"Number of points matched in all four images: {len(points_3d)}")

    if len(points_3d) < 4:
        raise RuntimeError("Need at least 4 common points for PnP.")

    np.random.seed(0)
    indices = np.random.choice(len(points_3d), 4, replace=False)
    print(f"PnP sampled indices: {indices}")

    T_left0_to_left1 = solve_pnp_safe(
        points_3d[indices],
        left1[indices],
        K,
        cv2.SOLVEPNP_EPNP,
    )

    if T_left0_to_left1 is None:
        raise RuntimeError("PnP failed.")

    print("Estimated extrinsic matrix [R | t]:")
    print(np.round(T_left0_to_left1, 4))

    plot_task_3_3(T_left0_to_left1, P1, P2)

    return T_left0_to_left1


def section_3_4(
    point_cloud_0: StereoPointCloud,
    point_cloud_1: StereoPointCloud,
    left0_pts: np.ndarray,
    left1_pts: np.ndarray,
    T_left0_to_left1: np.ndarray,
    threshold: float = SUPPORTER_THRESHOLD_PIXELS,
) -> np.ndarray:
    """Counts and plots supporters of the transformation estimated in Section 3.3."""
    print("--- Section 3.4 ---")

    P1, P2 = read_stereo_calibration()
    K = P1[:, :3]

    correspondences = find_common_points(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
    )

    points_3d = correspondences.points_3d
    left0 = correspondences.left0
    right0 = correspondences.right0
    left1 = correspondences.left1
    right1 = correspondences.right1

    supporter_mask, errors = count_supporters(
        T_left0_to_left1,
        points_3d,
        left0,
        right0,
        left1,
        right1,
        K,
        P1,
        P2,
        threshold,
    )

    num_supporters = int(np.sum(supporter_mask))
    total_points = len(supporter_mask)
    supporter_percentage = 100.0 * num_supporters / total_points if total_points > 0 else 0.0

    print(f"Number of four-view matches: {total_points}")
    print(f"Number of supporters: {num_supporters}")
    print(f"Supporter percentage: {supporter_percentage:.2f}%")

    plot_task_3_4(
        point_cloud_0.data.left_img,
        point_cloud_1.data.left_img,
        left0,
        left1,
        supporter_mask,
    )

    return supporter_mask


def section_3_5(
    point_cloud_0: StereoPointCloud,
    point_cloud_1: StereoPointCloud,
    left0_pts: np.ndarray,
    left1_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs RANSAC with PnP as the inner model, refines the transformation using
    all inliers, and plots the final inliers/outliers and transformed point clouds.
    """
    print("--- Section 3.5 ---")

    P1, P2 = read_stereo_calibration()
    K = P1[:, :3]

    correspondences = find_common_points(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
    )

    points_3d = correspondences.points_3d
    left0 = correspondences.left0
    right0 = correspondences.right0
    left1 = correspondences.left1
    right1 = correspondences.right1

    T_ransac, inlier_mask = ransac_pnp(
        points_3d,
        left1,
        left0,
        right0,
        right1,
        K,
        P1,
        P2,
    )

    if T_ransac is None or inlier_mask is None:
        raise RuntimeError("RANSAC failed to find a valid PnP transformation.")

    inlier_mask = np.asarray(inlier_mask, dtype=bool).flatten()
    num_inliers = int(np.sum(inlier_mask))
    num_outliers = len(inlier_mask) - num_inliers

    print(f"Number of RANSAC inliers: {num_inliers}")
    print(f"Number of RANSAC outliers: {num_outliers}")
    print(f"Inlier percentage: {100.0 * num_inliers / len(inlier_mask):.2f}%")

    T_refined = solve_pnp_safe(
        points_3d[inlier_mask],
        left1[inlier_mask],
        K,
        cv2.SOLVEPNP_EPNP,
    )

    if T_refined is None:
        print("Refinement failed; using the best RANSAC transformation instead.")
        T_refined = T_ransac

    print("Refined transformation T:")
    print(np.round(T_refined, 4))

    plot_task_3_5_matches(
        point_cloud_0.data.left_img,
        point_cloud_1.data.left_img,
        left0,
        left1,
        inlier_mask,
    )

    plot_task_3_5_point_clouds(point_cloud_0, point_cloud_1, T_refined, 300.0)

    return T_refined, inlier_mask


# def section_3_6() -> None:
#     print("--- Section 3.6 ---")
#
#     estimated_positions, relative_transforms, elapsed_time = track_sequence(
#         sequence_dir=SEQ_DIR,
#         num_frames=None,
#         feature_type=FEATURE_TYPE,
#         num_features=3000,
#         use_ratio_test=True,
#     )
#
#     gt_poses = read_ground_truth_poses(POSES_PATH)
#
#     print(f"Tracking took {elapsed_time:.2f} seconds")
#     print(f"Estimated {len(relative_transforms)} relative transformations")
#     plot_trajectory(estimated_positions, gt_poses)


def section_3_6() -> None:
    print("--- Section 3.6 ---")

    estimated_positions, relative_transforms, elapsed_time = track_sequence(
        sequence_dir=SEQ_DIR,
        num_frames=None,
        feature_type=FEATURE_TYPE,
        num_features=3000,
        use_ratio_test=True,
    )

    gt_poses = read_ground_truth_poses(POSES_PATH)

    print(f"Tracking took {elapsed_time:.2f} seconds")
    print(f"Estimated {len(relative_transforms)} relative transformations")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the raw relative PnP transforms as well, for debugging/reproducibility.
    np.save(
        OUTPUT_DIR / "relative_transforms.npy",
        np.asarray(relative_transforms),
    )

    world_to_camera_extrinsics = compose_frame0_to_camera_extrinsics(relative_transforms)
    camera_centers = camera_centers_from_world_to_camera_extrinsics(world_to_camera_extrinsics)

    np.save(
        OUTPUT_DIR / "global_camera_matrices.npy",
        world_to_camera_extrinsics,
    )
    np.save(OUTPUT_DIR / "camera_centers.npy", camera_centers)

    print(
        f"Saved global camera matrices with shape "
        f"{camera_centers.shape} to "
        f"{OUTPUT_DIR / 'global_camera_matrices.npy'}"
    )

    plot_trajectory(estimated_positions, gt_poses)


def main() -> None:
    point_cloud_0, point_cloud_1 = section_3_1()

    left0_pts, left1_pts, _ = section_3_2(
        point_cloud_0,
        point_cloud_1,
    )

    T_left0_to_left1 = section_3_3(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
    )

    section_3_4(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
        T_left0_to_left1,
    )

    section_3_5(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
    )

    section_3_6()


if __name__ == "__main__":
    main()
