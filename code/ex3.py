import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.image_loader import read_images
from utils.matching import extract_and_match_features, get_matched_points
from utils.visualization import plot_point_cloud_on_axis
from utils.stereo_pipeline import StereoMatchData, StereoPointCloud, DEVIATION_THRESHOLD, compute_rejection_statistics
from utils.stereo_pipeline import create_stereo_point_cloud

FRAME_0_INDEX = 0
FRAME_1_INDEX = 1
NUM_TEMPORAL_MATCHES_TO_DRAW = 200


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


def plot_temporal_left_matches(
    left0_img: np.ndarray,
    kp_left0: list[cv2.KeyPoint],
    left1_img: np.ndarray,
    kp_left1: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    num_to_draw: int = NUM_TEMPORAL_MATCHES_TO_DRAW,
) -> None:
    """Plots feature matches between left_0 and left_1."""
    matches_to_draw = matches[: min(num_to_draw, len(matches))]

    if left0_img.ndim == 2:
        left0_vis = cv2.cvtColor(left0_img, cv2.COLOR_GRAY2RGB)
    else:
        left0_vis = cv2.cvtColor(left0_img, cv2.COLOR_BGR2RGB)

    if left1_img.ndim == 2:
        left1_vis = cv2.cvtColor(left1_img, cv2.COLOR_GRAY2RGB)
    else:
        left1_vis = cv2.cvtColor(left1_img, cv2.COLOR_BGR2RGB)

    h0, w0 = left0_vis.shape[:2]
    h1, w1 = left1_vis.shape[:2]

    canvas_width = max(w0, w1)
    canvas_height = h0 + h1

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=left0_vis.dtype)
    canvas[:h0, :w0] = left0_vis
    canvas[h0:h0 + h1, :w1] = left1_vis

    plt.figure(figsize=(10, 12))
    plt.imshow(canvas)
    plt.axis("off")
    plt.title("Temporal feature matches between left_0 and left_1")

    for match in matches_to_draw:
        x0, y0 = kp_left0[match.queryIdx].pt
        x1, y1 = kp_left1[match.trainIdx].pt

        # left1 is drawn below left0, so shift its y-coordinate by h0
        y1_shifted = y1 + h0

        plt.scatter([x0, x1], [y0, y1_shifted], s=20)
        plt.plot([x0, x1], [y0, y1_shifted], linewidth=1)

    plt.tight_layout()
    plt.show()


def section_3_2(
    frame_idx0: int = FRAME_0_INDEX,
    frame_idx1: int = FRAME_1_INDEX,
) -> tuple[np.ndarray, np.ndarray, list[cv2.DMatch]]:
    """
    Section 3.2:
    Matches features between the two left images: left_0 and left_1.
    """
    print("--- Section 3.2 ---")

    left0_img, _ = read_images(frame_idx0)
    left1_img, _ = read_images(frame_idx1)

    kp_left0, kp_left1, matches = extract_and_match_features(left0_img, left1_img)

    left0_pts, left1_pts = get_matched_points(kp_left0, kp_left1, matches)

    print(f"Number of ratio-test matches between left_{frame_idx0} and left_{frame_idx1}: {len(matches)}")

    plot_temporal_left_matches(
        left0_img,
        kp_left0,
        left1_img,
        kp_left1,
        matches,
    )

    return left0_pts, left1_pts, matches


def main() -> None:
    point_cloud_0, point_cloud_1 = section_3_1()
    left0_pts, left1_pts, temporal_matches = section_3_2()

    # These will be useful for the next sections:
    # point_cloud_0.left_inliers  - left image points in frame 0
    # point_cloud_0.points_3d     - triangulated 3D points from frame 0
    # point_cloud_1.left_inliers  - left image points in frame 1
    # point_cloud_1.points_3d     - triangulated 3D points from frame 1


if __name__ == "__main__":
    main()