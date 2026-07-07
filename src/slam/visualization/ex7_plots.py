from pathlib import Path
import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from src.slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic
from src.slam.loop_closure.candidate_detection import ConsensusMatchResult
from src.slam.config import SEQUENCE_DIR

def _load_left_image(frame_id: int) -> np.ndarray:
    image_path = Path(SEQUENCE_DIR) / "image_0" / f"{frame_id:06d}.png"
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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

def plot_consensus_match(result: ConsensusMatchResult, output_path: Path) -> None:
    """Plot the match results of a single successful consensus match."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img1 = _load_left_image(result.candidate.source_frame)
    img2 = _load_left_image(result.candidate.target_frame)
    
    keypoints1 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in result.source_left]
    keypoints2 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in result.target_left]
    
    matches_inlier = []
    matches_outlier = []
    
    for i, is_inlier in enumerate(result.inlier_mask):
        match = cv2.DMatch(i, i, 0)
        if is_inlier:
            matches_inlier.append(match)
        else:
            matches_outlier.append(match)
            
    # Draw outliers in red, inliers in cyan (as requested by instructions)
    img_outliers = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches_outlier, None,
        matchColor=(255, 0, 0), singlePointColor=(255, 0, 0), flags=0
    )
    img_matches = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches_inlier, img_outliers,
        matchColor=(0, 255, 255), singlePointColor=(0, 255, 255), flags=0
    )
    
    plt.figure(figsize=(15, 6))
    plt.imshow(img_matches)
    plt.title(f"Consensus Match: c_{result.candidate.source_frame} -> c_{result.candidate.target_frame} (Inliers in Cyan, Outliers in Red)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def _add_covariance_ellipses(axis, marginals, frame_ids, estimated_positions, step=10):
    for index, frame_id in enumerate(frame_ids):
        if index % step != 0:
            continue

        covariance = marginals.marginalCovariance(gtsam.symbol("c", frame_id))
        translation_covariance = covariance[3:6, 3:6]
        xz_covariance = np.array([
            [translation_covariance[0, 0], translation_covariance[0, 2]],
            [translation_covariance[2, 0], translation_covariance[2, 2]],
        ])

        eigenvalues, eigenvectors = np.linalg.eigh(xz_covariance)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 10 * 2 * 3 * np.sqrt(np.maximum(eigenvalues, 1e-9))

        ellipse = Ellipse(
            xy=(estimated_positions[index, 0], estimated_positions[index, 2]),
            width=width, height=height, angle=angle,
            edgecolor="red", facecolor="none", alpha=0.5,
        )
        axis.add_patch(ellipse)
        
    axis.plot([], [], color="red", label="3-Sigma Covariance")

def plot_pose_graphs_versions(versions: list, frame_ids: list[int], output_path: Path) -> None:
    """Plot 4 versions of the pose graph along the process."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (title, values, marginals) in enumerate(versions):
        ax = axes[i]
        positions = positions_from_values(values, frame_ids)
        ax.plot(positions[:, 0], positions[:, 2], label="Pose Graph", color="blue", linewidth=2)
        
        if marginals is not None:
            _add_covariance_ellipses(ax, marginals, frame_ids, positions, step=10)
            
        ax.set_title(title)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()
        
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_pose_graph_comparisons(
    values_no_lc: gtsam.Values,
    values_lc: gtsam.Values,
    gt_extrinsics: np.ndarray,
    frame_ids: list[int],
    output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pos_no_lc = positions_from_values(values_no_lc, frame_ids)
    pos_lc = positions_from_values(values_lc, frame_ids)
    pos_gt = positions_from_world_to_camera_extrinsics(gt_extrinsics, frame_ids)
    
    plt.figure(figsize=(10, 8))
    plt.plot(pos_gt[:, 0], pos_gt[:, 2], label="Ground Truth", color="black", linestyle="--")
    plt.plot(pos_no_lc[:, 0], pos_no_lc[:, 2], label="Without Loop Closures", color="red", alpha=0.7)
    plt.plot(pos_lc[:, 0], pos_lc[:, 2], label="With Loop Closures", color="blue", alpha=0.7)
    
    plt.title("Trajectory Comparison")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_absolute_location_error(
    values_no_lc: gtsam.Values,
    values_lc: gtsam.Values,
    gt_extrinsics: np.ndarray,
    frame_ids: list[int],
    output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pos_no_lc = positions_from_values(values_no_lc, frame_ids)
    pos_lc = positions_from_values(values_lc, frame_ids)
    pos_gt = positions_from_world_to_camera_extrinsics(gt_extrinsics, frame_ids)
    
    err_no_lc = np.linalg.norm(pos_no_lc - pos_gt, axis=1)
    err_lc = np.linalg.norm(pos_lc - pos_gt, axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(frame_ids, err_no_lc, label="Without Loop Closures", color="red")
    plt.plot(frame_ids, err_lc, label="With Loop Closures", color="blue")
    
    plt.title("Absolute Location Error")
    plt.xlabel("Frame ID")
    plt.ylabel("Error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_location_uncertainty(
    marginals_no_lc: gtsam.Marginals,
    marginals_lc: gtsam.Marginals,
    frame_ids: list[int],
    output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    uncert_no_lc = []
    uncert_lc = []
    
    for frame_id in frame_ids:
        cov_no_lc = marginals_no_lc.marginalCovariance(gtsam.symbol("c", frame_id))[3:6, 3:6]
        cov_lc = marginals_lc.marginalCovariance(gtsam.symbol("c", frame_id))[3:6, 3:6]
        
        # We use determinant as a measure of uncertainty size
        uncert_no_lc.append(np.linalg.det(cov_no_lc))
        uncert_lc.append(np.linalg.det(cov_lc))
        
    plt.figure(figsize=(10, 5))
    plt.plot(frame_ids, uncert_no_lc, label="Without Loop Closures", color="red")
    plt.plot(frame_ids, uncert_lc, label="With Loop Closures", color="blue")
    
    plt.title("Location Uncertainty Size (det(Cov_translation))")
    plt.xlabel("Frame ID")
    plt.ylabel("Determinant of Translation Covariance")
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
