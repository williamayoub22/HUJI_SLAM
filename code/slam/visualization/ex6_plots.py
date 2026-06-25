from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from slam.ba.gtsam_utils import (
    pose3_from_world_to_camera_extrinsic,
)


def positions_from_world_to_camera_extrinsics(
    world_to_camera_extrinsics: np.ndarray,
    frame_ids: list[int],
) -> np.ndarray:
    return np.vstack(
        [
            np.asarray(
                pose3_from_world_to_camera_extrinsic(
                    world_to_camera_extrinsics[frame_id],
                ).translation(),
                dtype=float,
            ).reshape(3)
            for frame_id in frame_ids
        ]
    )


def positions_from_values(
    values: gtsam.Values,
    frame_ids: list[int],
) -> np.ndarray:
    return np.vstack(
        [
            np.asarray(
                values.atPose3(
                    gtsam.symbol("c", frame_id),
                ).translation(),
                dtype=float,
            ).reshape(3)
            for frame_id in frame_ids
        ]
    )


def plot_pose_graph_top_down(
    estimated_positions: np.ndarray,
    output_path: Path,
    title: str,
    marginals: gtsam.Marginals | None = None,
    frame_ids: list[int] | None = None,
) -> None:
    """
    Reproduces the Ex6.2 plotting style from the original ex6.py.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(figsize=(10, 8))

    axis.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 2],
        label="Pose Graph",
        color="blue",
        linewidth=2,
    )


    if marginals is not None and frame_ids is not None:
        for index, frame_id in enumerate(frame_ids):
            # Same behavior as the original implementation.
            if index % 10 != 0:
                continue

            covariance = marginals.marginalCovariance(
                gtsam.symbol("c", frame_id),
            )

            translation_covariance = covariance[3:6, 3:6]

            xz_covariance = np.array(
                [
                    [
                        translation_covariance[0, 0],
                        translation_covariance[0, 2],
                    ],
                    [
                        translation_covariance[2, 0],
                        translation_covariance[2, 2],
                    ],
                ]
            )

            eigenvalues, eigenvectors = np.linalg.eigh(
                xz_covariance,
            )

            angle = np.degrees(
                np.arctan2(
                    eigenvectors[1, 0],
                    eigenvectors[0, 0],
                )
            )

            # Same 3-sigma scale and numerical floor as original ex6.py.
            width, height = (
                10
                * 2
                * 3
                * np.sqrt(
                    np.maximum(eigenvalues, 1e-9),
                )
            )

            ellipse = Ellipse(
                xy=(
                    estimated_positions[index, 0],
                    estimated_positions[index, 2],
                ),
                width=width,
                height=height,
                angle=angle,
                edgecolor="red",
                facecolor="none",
                alpha=0.5,
            )

            axis.add_patch(ellipse)

        axis.plot(
            [],
            [],
            color="red",
            label="3-Sigma Covariance",
        )

    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.axis("equal")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    print(f"Saving plot to: {output_path}")

    figure.savefig(
        output_path,
        dpi=200,
    )
    plt.close(figure)


def plot_bundle_window_trajectory_with_covariances(
    graph: gtsam.NonlinearFactorGraph,
    optimized_values: gtsam.Values,
    output_path: Path,
) -> None:
    """
    3D covariance plot for Ex6.1, as requested by the exercise.
    """
    from gtsam.utils import plot

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    marginals = gtsam.Marginals(
        graph,
        optimized_values,
    )

    figure = plt.figure(figsize=(10, 8))

    plot.plot_trajectory(
        figure.number,
        optimized_values,
        marginals=marginals,
        scale=1,
    )

    plt.tight_layout()

    print(f"Saving plot to: {output_path}")

    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close(figure)
