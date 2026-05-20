import matplotlib.pyplot as plt
import numpy as np

from utils.visualization import plot_point_cloud_on_axis
from utils.stereo_pipeline import StereoMatchData, StereoPointCloud, DEVIATION_THRESHOLD, compute_rejection_statistics
from utils.stereo_pipeline import create_stereo_point_cloud

FRAME_0_INDEX = 0
FRAME_1_INDEX = 1


def print_rejection_statistics(data: StereoMatchData) -> None:
    """Prints match rejection statistics for a vertical-deviation threshold."""
    num_matches, num_rejected, percentage_rejected = compute_rejection_statistics(data)

    print(f"Total matches: {num_matches}")
    print(f"Matches with vertical deviation > {DEVIATION_THRESHOLD} px: {num_rejected}")
    print(f"Percentage: {percentage_rejected:.2f}%")


def section_3_1(
    threshold: float = DEVIATION_THRESHOLD,
) -> tuple[StereoPointCloud, StereoPointCloud]:
    """
    Section 3.1:
    Creates point clouds for stereo pair 0 and stereo pair 1.

    Pair 0 is the point cloud from Exercise 2.
    Pair 1 is created from:
        left1:  VAN_ex/dataset/sequences/00/image_0/000001.png
        right1: VAN_ex/dataset/sequences/00/image_1/000001.png
    """
    print("--- Section 3.1 ---")

    point_cloud_0 = create_stereo_point_cloud(FRAME_0_INDEX, threshold)
    point_cloud_1 = create_stereo_point_cloud(FRAME_1_INDEX, threshold)

    print("\nFrame 0:")
    print_rejection_statistics(point_cloud_0.data)
    print(f"Number of triangulated points: {len(point_cloud_0.points_3d)}")

    print("\nFrame 1:")
    print_rejection_statistics(point_cloud_1.data)
    print(f"Number of triangulated points: {len(point_cloud_1.points_3d)}")

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


def main() -> None:
    point_cloud_0, point_cloud_1 = section_3_1()

    # These will be useful for the next sections:
    # point_cloud_0.left_inliers  - left image points in frame 0
    # point_cloud_0.points_3d     - triangulated 3D points from frame 0
    # point_cloud_1.left_inliers  - left image points in frame 1
    # point_cloud_1.points_3d     - triangulated 3D points from frame 1


if __name__ == "__main__":
    main()