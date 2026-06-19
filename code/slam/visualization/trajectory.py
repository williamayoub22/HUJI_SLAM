import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from gtsam.symbol_shorthand import C, Q

from slam.ba.window_selection import _bundle_camera_centers, _bundle_landmarks


def camera_centers_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """Computes camera centers C = -R^T t from 3x4 extrinsic matrices."""
    centers = []

    for P in extrinsic:
        R = P[:, :3]
        t = P[:, 3]
        centers.append(-R.T @ t)

    return np.asarray(centers)


def plot_trajectory(
    estimated_positions: np.ndarray,
    gt_extrinsic: np.ndarray,
) -> None:
    """Plots estimated and ground-truth camera trajectories from above."""
    gt_positions = camera_centers_from_extrinsic(gt_extrinsic)

    plt.figure(figsize=(10, 8))

    plt.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 2],
        label="Estimated",
    )

    plt.plot(
        gt_positions[: len(estimated_positions), 0],
        gt_positions[: len(estimated_positions), 2],
        label="Ground Truth",
    )

    plt.title("Camera Trajectory Viewed from Above")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bundle_scene_3d(result, output_path: Path) -> None:
    """Plot optimized camera centers and landmarks in 3D."""
    camera_centers = _bundle_camera_centers(result)
    landmarks = _bundle_landmarks(result)

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    axis.plot(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        color="tab:red",
        marker="o",
        markersize=6,
        linewidth=2.5,
        label="Camera trajectory",
        zorder=10,
    )

    axis.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        landmarks[:, 2],
        s=4,
        color="tab:blue",
        alpha=0.28,
        label="Landmarks",
    )

    for frame_id, center in zip(result.window_frames, camera_centers):
        axis.text(center[0], center[1], center[2], str(frame_id), fontsize=8)

    axis.set_title("Optimized local bundle: cameras and landmarks")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")

    # Keep the local bundle scene readable by focusing on the near field.
    axis.set_xlim(-15, 10)
    axis.set_ylim(-8, 10)
    axis.set_zlim(0, 80)

    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_bundle_scene_top_down(result, output_path: Path) -> None:
    """Plot optimized camera centers and landmarks in the x-z plane."""
    camera_centers = _bundle_camera_centers(result)
    landmarks = _bundle_landmarks(result)

    figure, axis = plt.subplots(figsize=(10, 7))

    axis.plot(
        camera_centers[:, 0],
        camera_centers[:, 2],
        marker="o",
        color="red",
        label="Camera centers",
    )

    if len(landmarks) > 0:
        axis.scatter(
            landmarks[:, 0],
            landmarks[:, 2],
            s=5,
            alpha=0.45,
            label="Landmarks",
        )

    for frame_id, center in zip(result.window_frames, camera_centers):
        axis.annotate(str(frame_id), (center[0], center[2]), fontsize=8)

    axis.set_title("Optimized local bundle: top-down view")
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
