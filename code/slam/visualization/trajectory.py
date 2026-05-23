import matplotlib.pyplot as plt
import numpy as np


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
