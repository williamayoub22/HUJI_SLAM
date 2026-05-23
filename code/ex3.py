import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam.visualization.ex3_plots import plot_four_image_matches, plot_task_3_3, plot_task_3_4
from slam.features.detectors import DEFAULT_ORB_NUM_FEATURES
from slam.pipeline.temporal_pipeline import match_left_frames
from slam.features.detectors import FeatureType
from slam.pipeline.stereo_pipeline import create_stereo_point_cloud, StereoPointCloud
from slam.visualization.visualization import plot_point_cloud_on_axis
from slam.geometry.pnp import solve_pnp_safe
from slam.geometry.correspondences import find_common_points
from slam.io.calibration import read_calib
from slam.geometry.projection import count_supporters
from slam.geometry.correspondences import find_common_points

FRAME_0_INDEX = 0
FRAME_1_INDEX = 1

COMMON_POINT_TOLERANCE = 1e-3
SUPPORTER_THRESHOLD_PIXELS = 2.0
PNP_NUM_POINTS = 4
FEATURE_TYPE: FeatureType = "orb"


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
        feature_type="orb",
        num_features=3000,
        use_ratio_test=True,
    )

    point_cloud_1 = create_stereo_point_cloud(
        FRAME_1_INDEX,
        reject_negative_depth=True,
        feature_type = "orb",
        num_features = 3000,
        use_ratio_test=True
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

    P1, P2 = read_calib()
    K = P1[:, :3]

    pts_3d, pts_l1_c, pts_l0_c, pts_r0_c, pts_r1_c = find_common_points(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
    )

    print(f"Number of points matched in all four images: {len(pts_3d)}")

    if len(pts_3d) < 4:
        raise RuntimeError("Need at least 4 common points for PnP.")

    indices = np.random.choice(len(pts_3d), 4, replace=False)

    T_left0_to_left1 = solve_pnp_safe(
        pts_3d[indices],
        pts_l1_c[indices],
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

    P1, P2 = read_calib()
    K = P1[:, :3]

    pts_3d, pts_l1_c, pts_l0_c, pts_r0_c, pts_r1_c = find_common_points(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
    )

    supporter_mask, errors = count_supporters(
        T_left0_to_left1,
        pts_3d,
        pts_l0_c,
        pts_r0_c,
        pts_l1_c,
        pts_r1_c,
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
        pts_l0_c,
        pts_l1_c,
        supporter_mask,
    )

    return supporter_mask


def main() -> None:
    point_cloud_0, point_cloud_1 = section_3_1()

    left0_pts, left1_pts, temporal_matches = section_3_2(
        point_cloud_0,
        point_cloud_1,
    )

    T_left0_to_left1 = section_3_3(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
    )

    supporter_mask = section_3_4(
        point_cloud_0,
        point_cloud_1,
        left0_pts,
        left1_pts,
        T_left0_to_left1,
    )

if __name__ == "__main__":
    main()