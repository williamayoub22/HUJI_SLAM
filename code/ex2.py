from dataclasses import dataclass
from typing import List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.image_loader import load_matches_between_images
from utils.matching import get_matched_points
from utils.read_cam_calib import read_calib
from utils.triangulation import custom_triangulation


FRAME_INDEX = 0
VERTICAL_DEVIATION_THRESHOLD = 2.0
DEFAULT_FRAMES = (0, 1, 2, 3)


@dataclass
class StereoMatchData:
    """Container for one stereo pair, its matches, and vertical deviations."""
    left_img: np.ndarray
    right_img: np.ndarray
    kp_left: List[cv2.KeyPoint]
    kp_right: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]
    left_pts: np.ndarray
    right_pts: np.ndarray
    deviations: np.ndarray


def load_frame_data(frame_idx: int) -> StereoMatchData:
    """Loads one stereo frame and precomputes matched points and deviations."""
    kp_left, kp_right, left_img, matches, right_img = load_matches_between_images(frame_idx)
    left_pts, right_pts = get_matched_points(kp_left, kp_right, matches)
    deviations = np.abs(left_pts[:, 1] - right_pts[:, 1])

    return StereoMatchData(
        left_img=left_img,
        right_img=right_img,
        kp_left=kp_left,
        kp_right=kp_right,
        matches=matches,
        left_pts=left_pts,
        right_pts=right_pts,
        deviations=deviations,
    )


def get_inlier_points(
    data: StereoMatchData,
    threshold: float = VERTICAL_DEVIATION_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns matched points that satisfy the vertical-deviation threshold."""
    inlier_mask = data.deviations <= threshold
    return data.left_pts[inlier_mask], data.right_pts[inlier_mask]


def triangulate_opencv(
    P1: np.ndarray,
    P2: np.ndarray,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
) -> np.ndarray:
    """Triangulates matched points using OpenCV and converts to 3D coordinates."""
    points_4d = cv2.triangulatePoints(P1, P2, left_pts.T, right_pts.T)
    return (points_4d[:3, :] / points_4d[3, :]).T


def plot_vertical_deviation_histogram(deviations: np.ndarray) -> None:
    """Plots a histogram of vertical deviations between stereo matches."""
    plt.figure(figsize=(8, 5))
    plt.hist(deviations, bins=50)
    plt.xlabel(r"Vertical deviation $|y_L - y_R|$ [pixels]")
    plt.ylabel("Number of matches")
    plt.title("Histogram of vertical deviations from rectified stereo pattern")
    plt.grid(True)
    plt.show()


def show_image(ax, image: np.ndarray, title: str) -> None:
    """Displays an image on a given axis."""
    cmap = "gray" if image.ndim == 2 else None
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")


def plot_matches_by_rejection(
    data: StereoMatchData,
    threshold: float = VERTICAL_DEVIATION_THRESHOLD,
) -> None:
    """Plots accepted and rejected stereo matches on the image pair."""
    inlier_mask = data.deviations <= threshold
    outlier_mask = ~inlier_mask

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    show_image(ax_left, data.left_img, "Left Image")
    show_image(ax_right, data.right_img, "Right Image")

    ax_left.scatter(data.left_pts[outlier_mask, 0], data.left_pts[outlier_mask, 1],
                    c="cyan", s=15, label="Rejected (Outliers)")
    ax_right.scatter(data.right_pts[outlier_mask, 0], data.right_pts[outlier_mask, 1], c="cyan", s=15)

    ax_left.scatter(data.left_pts[inlier_mask, 0], data.left_pts[inlier_mask, 1],
                    c="orange", s=15, label="Accepted (Inliers)")
    ax_right.scatter(data.right_pts[inlier_mask, 0], data.right_pts[inlier_mask, 1], c="orange", s=15)

    fig.legend(loc="lower center", ncol=2)
    plt.suptitle(f"Rectified Stereo Match Rejection (Threshold: {threshold} px)")
    plt.tight_layout()
    plt.show()


def plot_point_cloud_on_axis(
    ax,
    points_3d: np.ndarray,
    title: str,
    color: str = "tab:blue",
) -> None:
    """Plots a 3D point cloud on a given axis."""
    ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        s=10,
        c=color,
        alpha=0.6,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=-70, azim=-90)


def print_rejection_statistics(
    data: StereoMatchData,
    threshold: float = VERTICAL_DEVIATION_THRESHOLD,
) -> None:
    """Prints match rejection statistics for a vertical-deviation threshold."""
    num_matches = len(data.matches)
    num_rejected = int(np.sum(data.deviations > threshold))
    percentage_rejected = 100.0 * num_rejected / num_matches

    print(f"Total matches: {num_matches}")
    print(f"Matches with vertical deviation > {threshold} px: {num_rejected}")
    print(f"Percentage: {percentage_rejected:.2f}%")


def section_2_1(frame_idx: int = FRAME_INDEX) -> StereoMatchData:
    print(f"--- Section 2.1 ---")

    data = load_frame_data(frame_idx)

    plot_vertical_deviation_histogram(data.deviations)
    print_rejection_statistics(data)

    return data


def section_2_2(
    data: StereoMatchData,
    threshold: float = VERTICAL_DEVIATION_THRESHOLD,
) -> int:
    print(f"--- Section 2.2 ---")

    num_discarded = int(np.sum(data.deviations > threshold))

    print(f"Total matches: {len(data.matches)}")
    print(f"Matches discarded (outliers): {num_discarded}\n")

    plot_matches_by_rejection(data, threshold)

    return num_discarded


def section_2_3(
    data: StereoMatchData,
    threshold: float = VERTICAL_DEVIATION_THRESHOLD,
) -> None:
    print(f"--- Section 2.3 ---")

    P1, P2 = read_calib()
    left_inliers, right_inliers = get_inlier_points(data, threshold)

    custom_points = custom_triangulation(P1, P2, left_inliers, right_inliers)
    opencv_points = triangulate_opencv(P1, P2, left_inliers, right_inliers)

    fig = plt.figure(figsize=(16, 7))

    ax_custom = fig.add_subplot(1, 2, 1, projection="3d")
    plot_point_cloud_on_axis(ax_custom, custom_points, "Custom DLT Triangulation")

    ax_opencv = fig.add_subplot(1, 2, 2, projection="3d")
    plot_point_cloud_on_axis(
        ax_opencv,
        opencv_points,
        "OpenCV Triangulation",
        color="tab:orange",
    )

    plt.tight_layout()
    plt.show()

    distances = np.linalg.norm(custom_points - opencv_points, axis=1)
    median_distance = np.median(distances)

    print(f"Number of triangulated points: {len(left_inliers)}")
    print(f"Median distance between Custom and OpenCV 3D points: {median_distance:.4e}\n")


def section_2_4(
    frames_to_test: Sequence[int] = DEFAULT_FRAMES,
    threshold: float = VERTICAL_DEVIATION_THRESHOLD,
) -> None:
    print(f"--- Section 2.4 ---")

    P1, P2 = read_calib()
    fig = plt.figure(figsize=(16, 14))

    for subplot_idx, frame_idx in enumerate(frames_to_test[:4], start=1):
        data = load_frame_data(frame_idx)
        left_inliers, right_inliers = get_inlier_points(data, threshold)
        points_3d = triangulate_opencv(P1, P2, left_inliers, right_inliers)

        print(f"Frame {frame_idx}: {len(points_3d)} triangulated points")

        ax = fig.add_subplot(2, 2, subplot_idx, projection="3d")
        plot_point_cloud_on_axis(
            ax,
            points_3d,
            f"Triangulation Point Cloud (Frame {frame_idx})",
        )

    plt.tight_layout()
    plt.show()


def main() -> None:
    data = section_2_1()
    section_2_2(data)
    section_2_3(data)
    section_2_4()


if __name__ == "__main__":
    main()