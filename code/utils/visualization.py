import numpy as np


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
