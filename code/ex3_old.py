import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

from slam.geometry.pnp import solve_pnp_safe
from slam.geometry.projection import count_supporters
from slam.geometry.correspondences import find_common_points
from slam.geometry.ransac import ransac_pnp
from slam.io.image_loader import read_images
from slam.features.detectors import extract_features
from slam.features.matching import match_and_filter, get_matched_points
from slam.io.calibration import read_calib
from slam.pipeline.stereo_pipeline import create_stereo_point_cloud


FRAME_0, FRAME_1 = 0, 1
MAX_RANSAC_ITERS = 200  # Dynamic cap converges fast
RANSAC_CONFIDENCE = 0.99  # 'p' probability of success
SUPPORTER_THRESH = 2.0  # pixels
RATIO_THRESHOLD = 0.75  # Lowe's Ratio Test (slightly relaxed for ORB)
NUM_FEATURES = 3000  # ORB needs more features than SIFT to compensate
DEVIATION_THRESHOLD = 2.0  # vertical deviation threshold for stereo outliers
MAX_DEPTH = 300.0  # reject triangulated points farther than this (meters)
MAX_TRANSLATION = 30.0  # reject PnP results with ||t|| > this (meters per frame)
MIN_INLIERS = 6  # minimum RANSAC supporters to accept a result
FEATURE_TYPE = "orb"
USE_RATIO_TEST = True

SEQ_DIR = Path(__file__).resolve().parent.parent / "dataset" / "sequences" / "00"
POSES_PATH = Path(__file__).resolve().parent.parent / "dataset" / "poses" / "00.txt"


def get_calib():
    P1, P2 = read_calib()
    return P1[:, :3], P1, P2


def track_sequence():
    """Runs the full tracking pipeline with Caching, ORB, & Lowe's Ratio Test."""
    K, P1, P2 = get_calib()
    # num_frames = len(list((SEQ_DIR / "image_0").glob("*.png")))
    num_frames = 200

    global_T = np.eye(4)
    trajectory = [np.zeros(3)]

    t0 = time.time()

    # Initialize "previous" state using Frame 0
    pc_prev = create_stereo_point_cloud(
        0,
        threshold=DEVIATION_THRESHOLD,
        reject_negative_depth=True,
        max_depth=MAX_DEPTH,
        feature_type=FEATURE_TYPE,
        num_features=NUM_FEATURES,
        use_ratio_test=USE_RATIO_TEST,
    )
    img_prev, _ = read_images(0)
    kp_prev, des_prev = extract_features(img_prev, FEATURE_TYPE ,NUM_FEATURES)

    for i in tqdm(range(num_frames - 1), desc="Tracking Frames", unit="frame"):
        # 1. Load and compute the "current" frame (i+1)
        img_curr, _ = read_images(i + 1)
        kp_curr, des_curr = extract_features(img_curr, FEATURE_TYPE, NUM_FEATURES)

        # Create current point cloud here so we can use it to find right_1 points
        pc_curr = create_stereo_point_cloud(i + 1, threshold=DEVIATION_THRESHOLD, max_depth=MAX_DEPTH)

        # Safety check
        if des_prev is None or des_curr is None:
            pc_prev = pc_curr
            kp_prev, des_prev = kp_curr, des_curr
            trajectory.append(trajectory[-1].copy())
            continue

        # 2. Match temporal features and apply Lowe's Ratio Test
        good_matches = match_and_filter(
            des_prev,
            des_curr,
            feature_type=FEATURE_TYPE,
            ratio=RATIO_THRESHOLD,
        )

        if len(good_matches) < 4:
            pc_prev = pc_curr
            kp_prev, des_prev = kp_curr, des_curr
            trajectory.append(trajectory[-1].copy())
            continue

        # Extract 2D points from the filtered matches
        pts_l0, pts_l1 = get_matched_points(kp_prev, kp_curr, good_matches)

        # 3. Find 3D points and run PnP
        pts_3d, pts_l1_c, pts_l0_c, pts_r0_c, pts_r1_c = find_common_points(pc_prev, pc_curr, pts_l0, pts_l1)

        # Extract best_mask alongside T_rel so we can plot inliers vs outliers
        T_rel, best_mask = ransac_pnp(pts_3d, pts_l1_c, pts_l0_c, pts_r0_c, pts_r1_c, K, P1, P2,
                                      max_iters=MAX_RANSAC_ITERS,
                                      confidence=RANSAC_CONFIDENCE,
                                      supporter_thresh=SUPPORTER_THRESH,
                                      max_translation=MAX_TRANSLATION,
                                      min_inliers=MIN_INLIERS)

        # Generate assignment plots AND print report statistics for the very first frame pair
        if i == 0 and T_rel is not None:
            # Task 3.4: Single PnP hypothesis run
            idx = np.random.choice(len(pts_3d), 4, replace=False)
            T_single = solve_pnp_safe(pts_3d[idx], pts_l1_c[idx], K, cv2.SOLVEPNP_EPNP)
            mask_single = np.zeros(len(pts_3d), dtype=bool)
            if T_single is not None:
                mask_single, _ = count_supporters(
                    T_single, pts_3d, pts_l0_c, pts_r0_c, pts_l1_c, pts_r1_c, K, P1, P2, SUPPORTER_THRESH
                )

            # --- PRINT STATISTICS FOR REPORT ---
            tqdm.write("\n" + "=" * 50)
            tqdm.write("REPORT STATISTICS (FRAME 0 -> 1)")
            tqdm.write("=" * 50)
            tqdm.write(f"Task 3.1 - Pair 0 Point Cloud Size : {len(pc_prev.points_3d)} points")
            tqdm.write(f"Task 3.1 - Pair 1 Point Cloud Size : {len(pc_curr.points_3d)} points")
            tqdm.write(f"Task 3.2 - Ratio-filtered Temporal Matches: {len(good_matches)} matches")
            tqdm.write(f"Task 3.3 - 4-Way Common Points    : {len(pts_3d)} matches given to PnP")
            if T_single is not None:
                tqdm.write(f"Task 3.4 - Single PnP Supporters  : {np.sum(mask_single)} / {len(pts_3d)}")
            tqdm.write(f"Task 3.5 - RANSAC Final Inliers   : {np.sum(best_mask)}")
            tqdm.write(f"Task 3.5 - RANSAC Final Outliers  : {len(pts_3d) - np.sum(best_mask)}")
            tqdm.write("=" * 50 + "\n")

            # --- PLOTS ---
            tqdm.write("Generating intermediate plots for Tasks 3.1 to 3.5...")
            plot_task_3_1_point_clouds(pc_prev, pc_curr)
            plot_task_3_2_temporal_matches(img_prev, img_curr, pts_l0, pts_l1)
            plot_task_3_3_cameras(T_rel, P1, P2)
            if T_single is not None:
                plot_task_3_4_single_pnp(img_prev, img_curr, pts_l0_c, pts_l1_c, mask_single)
            plot_task_3_5_ransac_matches(img_prev, img_curr, pts_l0_c, pts_l1_c, best_mask)
            plot_task_3_5_point_clouds(pc_prev, pc_curr, T_rel)

        if T_rel is None:
            T_rel = np.hstack((np.eye(3), np.zeros((3, 1))))

        step_T = np.eye(4)
        step_T[:3, :] = T_rel
        global_T = step_T @ global_T

        # Calculate position: -R^T * t
        pos = -global_T[:3, :3].T @ global_T[:3, 3]
        trajectory.append(pos)

        # --- SHIFT THE CACHE ---
        pc_prev = pc_curr
        kp_prev = kp_curr
        des_prev = des_curr
        img_prev = img_curr

    print(f"\nTracking took {time.time() - t0:.2f}s")
    return np.array(trajectory)


# ########################################################################### #
# ######################## EXERCISE PLOT FUNCTIONS ########################## #
# ########################################################################### #


def plot_trajectory(trajectory):
    """Plots estimated vs ground truth trajectory."""
    gt_poses = [np.array([float(x) for x in line.split()]).reshape(3, 4)
                for line in open(POSES_PATH)]
    gt_traj = np.array([-p[:3, :3].T @ p[:3, 3] for p in gt_poses])

    plt.figure(figsize=(10, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 2], 'b-', label='Estimated')
    plt.plot(gt_traj[:len(trajectory), 0], gt_traj[:len(trajectory), 2], 'r-', label='Ground Truth')
    plt.title("Camera Trajectory (X-Z plane)")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_task_3_1_point_clouds(pc0, pc1):
    """Task 3.1: Plot the 3D point clouds for both stereo pairs (Pair 0 and Pair 1)."""
    fig = plt.figure(figsize=(16, 7))

    for idx, (pc, title, color) in enumerate(zip(
            [pc0, pc1],
            ["Pair 0", "Pair 1"],
            ["tab:blue", "tab:orange"]
    )):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        pts_3d = pc.points_3d

        # Crop meaningless points to keep the plot focused
        crop_mask = (pts_3d[:, 2] > 0) & (pts_3d[:, 2] < 100)
        pts_cropped = pts_3d[crop_mask]

        ax.scatter(
            pts_cropped[:, 0],
            pts_cropped[:, 1],
            pts_cropped[:, 2],
            s=10,
            c=color,
            alpha=0.6,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Task 3.1: Triangulation Point Cloud ({title})\n{len(pts_cropped)} points")

        # Match the viewing angle from Exercise 2
        ax.view_init(elev=-70, azim=-90)

    plt.tight_layout()
    plt.show()


def plot_task_3_2_temporal_matches(img0, img1, pts_l0, pts_l1):
    """Task 3.2: Plot the temporal matches between left_0 and left_1 with connecting lines."""
    h1, w1 = img0.shape[:2]
    h2, w2 = img1.shape[:2]

    # Create a combined side-by-side image
    combined_img = np.zeros((max(h1, h2), w1 + w2), dtype=img0.dtype)
    combined_img[:h1, :w1] = img0
    combined_img[:h2, w1:w1 + w2] = img1

    plt.figure(figsize=(16, 6))
    plt.imshow(combined_img, cmap='gray')

    # Shift x-coordinates for the points on the second image
    pts_l1_shifted = pts_l1.copy()
    pts_l1_shifted[:, 0] += w1

    # Draw connecting lines with reduced opacity (alpha=0.1)
    for p0, p1 in zip(pts_l0, pts_l1_shifted):
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], c='cyan', alpha=0.1, linewidth=0.5)

    # Draw points on top of the lines (zorder=5 pushes them to the front)
    plt.scatter(pts_l0[:, 0], pts_l0[:, 1], c="tab:blue", s=15, label="left_0 Matches", zorder=5)
    plt.scatter(pts_l1_shifted[:, 0], pts_l1_shifted[:, 1], c="tab:orange", s=15, label="left_1 Matches", zorder=5)

    plt.title(f"Task 3.2: Matches between left images ({len(pts_l0)} total matches)")
    plt.axis("off")
    plt.legend(loc="lower center", ncol=2)
    plt.tight_layout()
    plt.show()


def plot_task_3_3_cameras(T, P1, P2):
    """Task 3.3: Plot the relative position of the four cameras (from above)."""
    # Calculate baseline translation from projection matrices
    # P2[0,3] = K[0,0] * tx -> tx = P2[0,3] / P1[0,0]
    tx = P2[0, 3] / P1[0, 0]

    # Camera centers (C = -R^T * t)
    cam_l0 = np.array([0, 0, 0])
    cam_r0 = np.array([-tx, 0, 0])

    R, t = T[:3, :3], T[:3, 3]
    cam_l1 = -R.T @ t
    cam_r1 = cam_l1 + R.T @ np.array([-tx, 0, 0])

    plt.figure(figsize=(8, 8))

    # Plot stereo baselines (connecting left and right cameras)
    plt.plot([cam_l0[0], cam_r0[0]], [cam_l0[2], cam_r0[2]], 'k--', alpha=0.6, label='Stereo Baseline')
    plt.plot([cam_l1[0], cam_r1[0]], [cam_l1[2], cam_r1[2]], 'k--', alpha=0.6)

    # Plot camera locations
    plt.scatter(cam_l0[0], cam_l0[2], c='blue', s=200, marker='^', label='left_0')
    plt.scatter(cam_r0[0], cam_r0[2], c='cyan', s=200, marker='^', label='right_0')
    plt.scatter(cam_l1[0], cam_l1[2], c='red', s=200, marker='^', label='left_1')
    plt.scatter(cam_r1[0], cam_r1[2], c='orange', s=200, marker='^', label='right_1')

    # Add direction arrows pointing down the Z-axis of each camera
    plt.arrow(cam_l0[0], cam_l0[2], 0, 0.5, head_width=0.05, color='blue', alpha=0.5)
    plt.arrow(cam_r0[0], cam_r0[2], 0, 0.5, head_width=0.05, color='cyan', alpha=0.5)

    dir_1 = R.T @ np.array([0, 0, 1])  # Frame 1's Z-axis in Frame 0's coordinates
    plt.arrow(cam_l1[0], cam_l1[2], dir_1[0] * 0.5, dir_1[2] * 0.5, head_width=0.05, color='red', alpha=0.5)
    plt.arrow(cam_r1[0], cam_r1[2], dir_1[0] * 0.5, dir_1[2] * 0.5, head_width=0.05, color='orange', alpha=0.5)

    plt.title("Task 3.3: Relative Positions of 4 Cameras (Top-Down X-Z)")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_task_3_4_single_pnp(img0, img1, pts_l0, pts_l1, mask):
    """Task 3.4: Plot on images left_0 and left_1 the supporters for a SINGLE PnP hypothesis."""
    mask = np.array(mask, dtype=bool).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(img0, cmap='gray')
    axes[0].set_title("Temporal Matches on left_0")
    axes[1].imshow(img1, cmap='gray')
    axes[1].set_title("Temporal Matches on left_1")

    inlier_idx = np.where(mask)[0]
    outlier_idx = np.where(~mask)[0]

    # Outliers (Cyan)
    axes[0].scatter(pts_l0[outlier_idx, 0], pts_l0[outlier_idx, 1], c="cyan", s=15, label="Rejected (Outliers)")
    axes[1].scatter(pts_l1[outlier_idx, 0], pts_l1[outlier_idx, 1], c="cyan", s=15, label="Rejected (Outliers)")

    # Inliers/Supporters (Orange)
    axes[0].scatter(pts_l0[inlier_idx, 0], pts_l0[inlier_idx, 1], c="orange", s=15, label="Accepted (Inliers)")
    axes[1].scatter(pts_l1[inlier_idx, 0], pts_l1[inlier_idx, 1], c="orange", s=15, label="Accepted (Inliers)")

    for ax in axes:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)

    plt.suptitle("Task 3.4: Supporters for a Single Random PnP Hypothesis (4 Points)")
    plt.tight_layout()
    plt.show()


def plot_task_3_5_ransac_matches(img0, img1, pts_l0, pts_l1, mask):
    """Task 3.5: Plot on images left_0 and left_1 the FINAL RANSAC inliers and outliers."""
    mask = np.array(mask, dtype=bool).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(img0, cmap='gray')
    axes[0].set_title("Temporal Matches on left_0")
    axes[1].imshow(img1, cmap='gray')
    axes[1].set_title("Temporal Matches on left_1")

    inlier_idx = np.where(mask)[0]
    outlier_idx = np.where(~mask)[0]

    # Outliers (Cyan)
    axes[0].scatter(pts_l0[outlier_idx, 0], pts_l0[outlier_idx, 1], c="cyan", s=15, label="Rejected (Outliers)")
    axes[1].scatter(pts_l1[outlier_idx, 0], pts_l1[outlier_idx, 1], c="cyan", s=15, label="Rejected (Outliers)")

    # Inliers/Supporters (Orange)
    axes[0].scatter(pts_l0[inlier_idx, 0], pts_l0[inlier_idx, 1], c="orange", s=15, label="Accepted (Inliers)")
    axes[1].scatter(pts_l1[inlier_idx, 0], pts_l1[inlier_idx, 1], c="orange", s=15, label="Accepted (Inliers)")

    for ax in axes:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)

    plt.suptitle("Task 3.5: Final Best RANSAC Inliers vs Outliers")
    plt.tight_layout()
    plt.show()


def plot_task_3_5_point_clouds(pc0, pc1, T):
    """Task 3.5: Plot the two 3D point clouds (pair 1 and pair 0 after T)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Transform pc0 into left_1 coordinates: X_1 = R * X_0 + t
    R, t = T[:3, :3], T[:3, 3]
    pts0_transformed = (R @ pc0.points_3d.T).T + t
    pts1_3d = pc1.points_3d

    # Crop meaningless/infinity points to prevent the plot from stretching to infinity
    crop_mask0 = (pts0_transformed[:, 2] > 0) & (pts0_transformed[:, 2] < 100)
    crop_mask1 = (pts1_3d[:, 2] > 0) & (pts1_3d[:, 2] < 100)

    p0_cropped = pts0_transformed[crop_mask0]
    p1_cropped = pts1_3d[crop_mask1]

    ax.scatter(
        p0_cropped[:, 0],
        p0_cropped[:, 1],
        p0_cropped[:, 2],
        s=10,
        c="tab:blue",
        alpha=0.6,
        label="Pair 0 (Transformed)"
    )

    ax.scatter(
        p1_cropped[:, 0],
        p1_cropped[:, 1],
        p1_cropped[:, 2],
        s=10,
        c="tab:orange",
        alpha=0.6,
        label="Pair 1"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Task 3.5: 3D Point Clouds in left_1 Frame")

    # Set the same viewing angle as the previous exercise
    ax.view_init(elev=-70, azim=-90)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    start_time = time.time()

    traj = track_sequence()
    plot_trajectory(traj)

    elapsed_seconds = time.time() - start_time
    elapsed_minutes = int(elapsed_seconds // 60)
    remaining_seconds = elapsed_seconds % 60

    print(f"\nTotal execution time: {elapsed_minutes} minutes and {remaining_seconds:.2f} seconds.")