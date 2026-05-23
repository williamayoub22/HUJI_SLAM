import cv2
import numpy as np
from matplotlib import pyplot as plt

from ..pipeline.stereo_pipeline import StereoPointCloud
from ..pipeline.temporal_pipeline import TemporalMatchData

NUM_TEMPORAL_MATCHES_TO_DRAW = 200

def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Converts a grayscale/BGR image to RGB for matplotlib."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_four_image_matches(
    point_cloud_0: StereoPointCloud,
    point_cloud_1: StereoPointCloud,
    temporal_data: TemporalMatchData,
    num_to_draw: int = NUM_TEMPORAL_MATCHES_TO_DRAW,
) -> None:
    """
    Plots the four images in a 2x2 grid:
        left_0   right_0
        left_1   right_1

    Stereo matches are drawn in blue, and temporal left_0-left_1 matches are
    drawn in red.
    """
    left0_vis = _to_rgb(point_cloud_0.data.left_img)
    right0_vis = _to_rgb(point_cloud_0.data.right_img)
    left1_vis = _to_rgb(point_cloud_1.data.left_img)
    right1_vis = _to_rgb(point_cloud_1.data.right_img)

    h, w = left0_vis.shape[:2]

    canvas = np.zeros((2 * h, 2 * w, 3), dtype=left0_vis.dtype)
    canvas[:h, :w] = left0_vis
    canvas[:h, w:2 * w] = right0_vis
    canvas[h:2 * h, :w] = left1_vis
    canvas[h:2 * h, w:2 * w] = right1_vis

    plt.figure(figsize=(16, 10))
    plt.imshow(canvas)
    plt.axis("off")
    plt.title("Stereo matches and temporal matches")

    plt.text(10, 25, "left_0", color="white", fontsize=14, weight="bold")
    plt.text(w + 10, 25, "right_0", color="white", fontsize=14, weight="bold")
    plt.text(10, h + 25, "left_1", color="white", fontsize=14, weight="bold")
    plt.text(w + 10, h + 25, "right_1", color="white", fontsize=14, weight="bold")

    # Stereo matches for pair 0: left_0 -> right_0.
    stereo0_count = min(num_to_draw, len(point_cloud_0.left_inliers))
    for left_pt, right_pt in zip(
        point_cloud_0.left_inliers[:stereo0_count],
        point_cloud_0.right_inliers[:stereo0_count],
    ):
        x0, y0 = left_pt
        xr, yr = right_pt
        plt.plot(
            [x0, xr + w],
            [y0, yr],
            color="tab:blue",
            alpha=0.25,
            linewidth=0.7,
        )

    # Stereo matches for pair 1: left_1 -> right_1.
    stereo1_count = min(num_to_draw, len(point_cloud_1.left_inliers))
    for left_pt, right_pt in zip(
        point_cloud_1.left_inliers[:stereo1_count],
        point_cloud_1.right_inliers[:stereo1_count],
    ):
        x0, y0 = left_pt
        xr, yr = right_pt
        plt.plot(
            [x0, xr + w],
            [y0 + h, yr + h],
            color="tab:blue",
            alpha=0.25,
            linewidth=0.7,
        )

    # Temporal matches: left_0 -> left_1.
    temporal_count = min(num_to_draw, len(temporal_data.left0_pts))
    for left0_pt, left1_pt in zip(
        temporal_data.left0_pts[:temporal_count],
        temporal_data.left1_pts[:temporal_count],
    ):
        x0, y0 = left0_pt
        x1, y1 = left1_pt
        plt.plot(
            [x0, x1],
            [y0, y1 + h],
            color="tab:red",
            alpha=0.4,
            linewidth=0.8,
        )

    stereo_proxy = plt.Line2D([0], [0], color="tab:blue", linewidth=2)
    temporal_proxy = plt.Line2D([0], [0], color="tab:red", linewidth=2)

    plt.legend(
        [stereo_proxy, temporal_proxy],
        ["Stereo matches", "Temporal left matches"],
        loc="lower center",
        ncol=2,
    )

    plt.tight_layout()
    plt.show()

def plot_task_3_3(T, P1, P2):
    """Plots the relative positions of the four camera centers from above."""
    # P2[0, 3] = K[0, 0] * tx, so tx is the right-camera translation
    # relative to the left camera in the stereo calibration.
    tx = P2[0, 3] / P1[0, 0]
    stereo_baseline_left_to_right = np.array([-tx, 0.0, 0.0])

    # Camera centers in the left_0 coordinate system.
    cam_l0 = np.array([0.0, 0.0, 0.0])
    cam_r0 = cam_l0 + stereo_baseline_left_to_right

    R, t = T[:3, :3], T[:3, 3]
    cam_l1 = -R.T @ t
    cam_r1 = cam_l1 + R.T @ stereo_baseline_left_to_right

    camera_positions = {
        "left_0": cam_l0,
        "right_0": cam_r0,
        "left_1": cam_l1,
        "right_1": cam_r1,
    }

    plt.figure(figsize=(8, 8))

    # Connect each stereo pair with a dashed line.
    plt.plot(
        [cam_l0[0], cam_r0[0]],
        [cam_l0[2], cam_r0[2]],
        "k--",
        alpha=0.6,
        label="Stereo baseline",
    )
    plt.plot(
        [cam_l1[0], cam_r1[0]],
        [cam_l1[2], cam_r1[2]],
        "k--",
        alpha=0.6,
    )

    colors = {
        "left_0": "tab:blue",
        "right_0": "tab:cyan",
        "left_1": "tab:red",
        "right_1": "tab:orange",
    }

    for label, center in camera_positions.items():
        plt.scatter(
            center[0],
            center[2],
            c=colors[label],
            s=120,
            marker="o",
            label=label,
        )
        plt.annotate(
            label,
            (center[0], center[2]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    plt.title("Task 3.3: Relative Positions of the Four Cameras (Top-Down X-Z)")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
