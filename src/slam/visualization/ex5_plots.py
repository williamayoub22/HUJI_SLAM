from pathlib import Path

import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from slam.analysis.bundle_diagnostics import LargestFactorDiagnostic


def _make_track_x_axis(frame_ids: list[int], use_track_index: bool):
    """Returns x-values and x-label for plotting either absolute frame ids
    or relative indices inside the selected track.
    """
    if use_track_index:
        return np.arange(len(frame_ids)), "Frame index in selected track"

    return np.asarray(frame_ids), "Frame id"


def plot_reprojection_errors_q5_1(
    frame_ids: list[int],
    reprojection_errors: np.ndarray,
    track_id: int,
    output_path: str | Path,
    use_track_index: bool = True,
) -> None:
    """Plots the stereo reprojection L2 error over the selected track."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values, x_label = _make_track_x_axis(frame_ids, use_track_index)
    valid = np.isfinite(reprojection_errors)

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        x_values[valid],
        reprojection_errors[valid],
        marker="o",
        linewidth=1.2,
    )
    plt.xlabel(x_label)
    plt.ylabel("Stereo reprojection error [pixels]")
    plt.title(f"Track #{track_id}: stereo reprojection error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_factor_errors_q5_1(
    frame_ids: list[int],
    factor_errors: np.ndarray,
    track_id: int,
    output_path: str | Path,
    use_track_index: bool = True,
) -> None:
    """Plots the GTSAM stereo factor error over the selected track."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values, x_label = _make_track_x_axis(frame_ids, use_track_index)
    valid = np.isfinite(factor_errors)

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        x_values[valid],
        factor_errors[valid],
        marker="o",
        linewidth=1.2,
    )
    plt.xlabel(x_label)
    plt.ylabel("Factor error [Normalized Squared Error]")
    plt.title(f"Track #{track_id}: GTSAM stereo factor error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _draw_measurement_and_projection(
    axis,
    image: np.ndarray,
    measurement_xy: tuple[float, float],
    projection_xy: tuple[float, float],
    title: str,
) -> None:
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    axis.scatter(
        *measurement_xy,
        marker="x",
        s=90,
        linewidths=2.5,
        color="red",
        label="Measurement",
    )

    axis.scatter(
        *projection_xy,
        marker="o",
        s=90,
        linewidths=2.0,
        facecolors="none",
        edgecolors="lime",
        label="Projection",
    )

    axis.set_title(title)
    axis.axis("off")


def plot_largest_factor_diagnostic(
    diagnostic: LargestFactorDiagnostic,
    left_image: np.ndarray,
    right_image: np.ndarray,
    output_path,
) -> None:
    measurement = diagnostic.measurement

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))

    _draw_measurement_and_projection(
        axes[0, 0],
        left_image,
        measurement_xy=(measurement.uL(), measurement.v()),
        projection_xy=(
            diagnostic.initial.projection.uL(),
            diagnostic.initial.projection.v(),
        ),
        title=(f"Initial left image\ndistance = {diagnostic.initial.left_distance_pixels:.2f}px"),
    )

    _draw_measurement_and_projection(
        axes[0, 1],
        right_image,
        measurement_xy=(measurement.uR(), measurement.v()),
        projection_xy=(
            diagnostic.initial.projection.uR(),
            diagnostic.initial.projection.v(),
        ),
        title=(f"Initial right image\ndistance = {diagnostic.initial.right_distance_pixels:.2f}px"),
    )

    _draw_measurement_and_projection(
        axes[1, 0],
        left_image,
        measurement_xy=(measurement.uL(), measurement.v()),
        projection_xy=(
            diagnostic.optimized.projection.uL(),
            diagnostic.optimized.projection.v(),
        ),
        title=(
            f"Optimized left image\ndistance = {diagnostic.optimized.left_distance_pixels:.2f}px"
        ),
    )

    _draw_measurement_and_projection(
        axes[1, 1],
        right_image,
        measurement_xy=(measurement.uR(), measurement.v()),
        projection_xy=(
            diagnostic.optimized.projection.uR(),
            diagnostic.optimized.projection.v(),
        ),
        title=(
            f"Optimized right image\ndistance = {diagnostic.optimized.right_distance_pixels:.2f}px"
        ),
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
    )

    figure.suptitle(
        "Largest initial-error projection factor\n"
        f"frame c={diagnostic.frame_id}, landmark q={diagnostic.track_id}",
        y=0.99,
    )

    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_keyframes_and_landmarks_top_down(
    keyframe_ids: list[int],
    global_keyframe_poses: dict[int, gtsam.Pose3],
    landmarks_global: np.ndarray,
    gt_positions: np.ndarray,
    output_path: Path,
) -> None:
    estimated_positions = np.vstack(
        [
            np.asarray(
                global_keyframe_poses[frame_id].translation(),
                dtype=float,
            ).reshape(3)
            for frame_id in keyframe_ids
        ]
    )

    figure, axis = plt.subplots(figsize=(11, 8))

    if len(landmarks_global) > 0:
        axis.scatter(
            landmarks_global[:, 0],
            landmarks_global[:, 2],
            s=3,
            alpha=0.25,
            label="Optimized landmarks",
        )

    axis.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 2],
        marker="o",
        linewidth=2,
        label="Estimated keyframes",
    )

    axis.plot(
        gt_positions[:, 0],
        gt_positions[:, 2],
        marker="x",
        linewidth=1.5,
        label="Ground-truth keyframes",
    )

    for frame_id, position in zip(keyframe_ids, estimated_positions):
        axis.annotate(
            str(frame_id),
            (position[0], position[2]),
            fontsize=7,
        )

    axis.set_title("All bundle-adjusted keyframes and landmarks: top-down view")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)
    axis.legend()

    # Set axis limits based on keyframe positions so the trajectories are
    # visible and not squashed by outlier landmarks
    all_x = np.concatenate([estimated_positions[:, 0], gt_positions[:, 0]])
    all_z = np.concatenate([estimated_positions[:, 2], gt_positions[:, 2]])
    margin = 0.1 * max(all_x.ptp(), all_z.ptp(), 1.0)
    axis.set_xlim(all_x.min() - margin, all_x.max() + margin)
    axis.set_ylim(all_z.min() - margin, all_z.max() + margin)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_keyframe_localization_error(
    keyframe_ids: list[int],
    estimated_positions: np.ndarray,
    gt_positions: np.ndarray,
    output_path: Path,
) -> np.ndarray:
    errors = np.linalg.norm(
        estimated_positions - gt_positions,
        axis=1,
    )

    figure, axis = plt.subplots(figsize=(10, 5))

    axis.plot(
        keyframe_ids,
        errors,
        marker="o",
    )

    axis.set_title("Keyframe localization error")
    axis.set_xlabel("Frame index")
    axis.set_ylabel("Position error [m]")
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return errors
