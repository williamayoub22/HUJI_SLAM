import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import math
import time

from utils.image_loader import read_images
from utils.matching import extract_and_match_features, get_matched_points
from utils.visualization import plot_point_cloud_on_axis
from utils.stereo_pipeline import StereoPointCloud, DEVIATION_THRESHOLD
from utils.stereo_pipeline import create_stereo_point_cloud
from utils.read_cam_calib import read_calib
from pathlib import Path

# ==========================================================
# HYPERPARAMETERS
# ==========================================================
FRAME_0_INDEX = 0
FRAME_1_INDEX = 1
NUM_TEMPORAL_MATCHES_TO_DRAW = 200

# RANSAC / Tracking Optimization Parameters
MAX_RANSAC_ITERATIONS = 200
RANSAC_CONFIDENCE = 0.99
MAX_RANSAC_POINTS = 200  # Limit the max points passed to RANSAC for time optimization
PNP_SAMPLE_SIZE = 4
SUPPORTER_THRESHOLD = 2.0  # pixels

# Dataset paths
SCRIPT_DIR = Path(__file__).resolve().parent
POSES_PATH = SCRIPT_DIR.parent / "dataset" / "poses" / "00.txt"
SEQ_DIR = SCRIPT_DIR.parent / "dataset" / "sequences" / "00"


# ==========================================================
# Shared Helpers
# ==========================================================

def get_k_and_p_matrices():
    """Returns K, P1, P2."""
    P1, P2 = read_calib()
    K = P1[:, :3]
    return K, P1, P2


def rodriguez_to_mat(rvec, tvec):
    """Converts Rodrigues rotation vector + translation to a 3x4 [R|t] matrix."""
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def find_common_points(point_cloud_0, pts_left0, pts_left1):
    """
    Finds points that appear in both the stereo point cloud of pair 0
    and the temporal left0-left1 matches.

    Assuming pts_left0 and pts_left1 are already sorted by match quality (descending),
    the returned arrays will preserve this order, enabling PROSAC sampling.

    Returns:
        pts_3d_common:     Nx3  – 3D coordinates from triangulation of pair 0
        pts_2d_l1_common:  Nx2  – pixel locations on left_1
        pts_2d_l0_common:  Nx2  – pixel locations on left_0
        pts_2d_r0_common:  Nx2  – pixel locations on right_0
    """
    # Build dictionary from left_0 stereo inlier coords -> index in point cloud
    stereo_dict = {}
    for j, pt in enumerate(point_cloud_0.left_inliers):
        key = (float(pt[0]), float(pt[1]))
        stereo_dict[key] = j

    pts_3d_common = []
    pts_2d_l1_common = []
    pts_2d_l0_common = []
    pts_2d_r0_common = []

    for i, pt_temporal in enumerate(pts_left0):
        key = (float(pt_temporal[0]), float(pt_temporal[1]))
        if key in stereo_dict:
            j = stereo_dict[key]
            pts_3d_common.append(point_cloud_0.points_3d[j])
            pts_2d_l1_common.append(pts_left1[i])
            pts_2d_l0_common.append(pt_temporal)
            pts_2d_r0_common.append(point_cloud_0.right_inliers[j])

    return (np.array(pts_3d_common), np.array(pts_2d_l1_common),
            np.array(pts_2d_l0_common), np.array(pts_2d_r0_common))


# ==========================================================
# SECTION 3.1
# ==========================================================
def section_3_1(threshold=DEVIATION_THRESHOLD):
    """Creates point clouds for stereo pair 0 and stereo pair 1."""
    print("--- Section 3.1 ---")

    point_cloud_0 = create_stereo_point_cloud(FRAME_0_INDEX, threshold)
    point_cloud_1 = create_stereo_point_cloud(FRAME_1_INDEX, threshold)

    fig = plt.figure(figsize=(16, 7))

    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_point_cloud_on_axis(ax0, point_cloud_0.points_3d,
                             "Point Cloud from Stereo Pair 0", color="tab:blue")

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_point_cloud_on_axis(ax1, point_cloud_1.points_3d,
                             "Point Cloud from Stereo Pair 1", color="tab:orange")

    plt.tight_layout()
    plt.show()

    return point_cloud_0, point_cloud_1


# ==========================================================
# SECTION 3.2
# ==========================================================
def plot_temporal_left_matches(left0_img, kp_left0, left1_img, kp_left1,
                               matches, num_to_draw=NUM_TEMPORAL_MATCHES_TO_DRAW):
    """Plots feature matches between left_0 and left_1."""
    matches_to_draw = matches[:min(num_to_draw, len(matches))]

    left0_vis = cv2.cvtColor(left0_img, cv2.COLOR_GRAY2RGB) if left0_img.ndim == 2 \
        else cv2.cvtColor(left0_img, cv2.COLOR_BGR2RGB)
    left1_vis = cv2.cvtColor(left1_img, cv2.COLOR_GRAY2RGB) if left1_img.ndim == 2 \
        else cv2.cvtColor(left1_img, cv2.COLOR_BGR2RGB)

    h0, w0 = left0_vis.shape[:2]
    h1, w1 = left1_vis.shape[:2]

    canvas = np.zeros((h0 + h1, max(w0, w1), 3), dtype=left0_vis.dtype)
    canvas[:h0, :w0] = left0_vis
    canvas[h0:h0 + h1, :w1] = left1_vis

    plt.figure(figsize=(10, 12))
    plt.imshow(canvas)
    plt.axis("off")
    plt.title("Temporal feature matches between left_0 and left_1")

    for match in matches_to_draw:
        x0, y0 = kp_left0[match.queryIdx].pt
        x1, y1 = kp_left1[match.trainIdx].pt
        y1_shifted = y1 + h0
        plt.scatter([x0, x1], [y0, y1_shifted], s=20)
        plt.plot([x0, x1], [y0, y1_shifted], linewidth=1)

    plt.tight_layout()
    plt.show()


def section_3_2(frame_idx0=FRAME_0_INDEX, frame_idx1=FRAME_1_INDEX):
    """Matches features between left_0 and left_1."""
    print("--- Section 3.2 ---")

    left0_img, _ = read_images(frame_idx0)
    left1_img, _ = read_images(frame_idx1)

    kp_left0, kp_left1, matches = extract_and_match_features(left0_img, left1_img)

    # Sort matches by distance (ascending) so higher quality matches are first.
    # This prepares the data for PROSAC sampling.
    matches = sorted(matches, key=lambda m: m.distance)

    pts_left0, pts_left1 = get_matched_points(kp_left0, kp_left1, matches)

    print(f"Number of matches between left_{frame_idx0} and left_{frame_idx1}: {len(matches)}")

    plot_temporal_left_matches(left0_img, kp_left0, left1_img, kp_left1, matches)

    return pts_left0, pts_left1, matches


# ==========================================================
# SECTION 3.3
# ==========================================================
def plot_camera_positions(R, tvec):
    """Plots the X-Z bird's-eye view of the 4 cameras."""
    P1, P2 = read_calib()
    focal_length = P1[0, 0]
    tx_pixels = P2[0, 3]
    baseline = -tx_pixels / focal_length

    cam_l0_pos = np.array([0, 0, 0])
    cam_r0_pos = np.array([baseline, 0, 0])

    cam_l1_pos = -R.T @ tvec.flatten()
    cam_r1_pos = cam_l1_pos + R.T @ np.array([baseline, 0, 0])

    plt.figure(figsize=(8, 6))
    cameras_x = [cam_l0_pos[0], cam_r0_pos[0], cam_l1_pos[0], cam_r1_pos[0]]
    cameras_z = [cam_l0_pos[2], cam_r0_pos[2], cam_l1_pos[2], cam_r1_pos[2]]
    labels = ['$left_0$', '$right_0$', '$left_1$', '$right_1$']
    colors = ['blue', 'blue', 'orange', 'orange']

    plt.scatter(cameras_x, cameras_z, c=colors, s=100, marker='s')
    for i, label in enumerate(labels):
        plt.annotate(label, (cameras_x[i], cameras_z[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    plt.plot([cam_l0_pos[0], cam_r0_pos[0]], [cam_l0_pos[2], cam_r0_pos[2]],
             'b--', label='Stereo Pair 0')
    plt.plot([cam_l1_pos[0], cam_r1_pos[0]], [cam_l1_pos[2], cam_r1_pos[2]],
             color='orange', linestyle='--', label='Stereo Pair 1')
    plt.plot([cam_l0_pos[0], cam_l1_pos[0]], [cam_l0_pos[2], cam_l1_pos[2]],
             'gray', linestyle=':', label='Camera Motion')

    plt.title("Relative Position of Four Cameras (Top-Down View)")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def section_3_3(point_cloud_0, pts_left0, pts_left1):
    """
    Finds 4 key-points matched on all four images and calculates [R|t] of left_1
    using PnP.
    """
    print("--- Section 3.3 ---")

    K, _, _ = get_k_and_p_matrices()

    pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common = \
        find_common_points(point_cloud_0, pts_left0, pts_left1)

    print(f"Found {len(pts_3d_common)} common points across all four images.")

    if len(pts_3d_common) < PNP_SAMPLE_SIZE:
        raise ValueError("Not enough common points to compute PnP.")

    # Choose 4 random key-points
    random_indices = numpy.random.choice(len(pts_3d_common), PNP_SAMPLE_SIZE, replace=False)
    sample_3d = pts_3d_common[random_indices].astype(np.float64)
    sample_2d = pts_2d_l1_common[random_indices].astype(np.float64)

    # Apply PnP
    success, rvec, tvec = cv2.solvePnP(sample_3d, sample_2d, K, None, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        raise RuntimeError("cv2.solvePnP failed.")

    R, _ = cv2.Rodrigues(rvec)
    T_mat = np.hstack((R, tvec))

    print(f"Calculated Extrinsic Matrix [R|t]:\n{np.round(T_mat, 4)}")

    plot_camera_positions(R, tvec)

    return pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common, T_mat


# ==========================================================
# SECTION 3.4 – Supporter counting
# ==========================================================
def project_points(points_3d, projection_matrix):
    """
    Projects Nx3 3D points using a 3x4 projection matrix.
    Returns Nx2 pixel coordinates.
    """
    ones = np.ones((points_3d.shape[0], 1))
    pts_h = np.hstack((points_3d, ones))  # Nx4
    projected = (projection_matrix @ pts_h.T).T  # Nx3
    return projected[:, :2] / projected[:, 2:3]


def count_supporters(T_left1, points_3d, pts_left0, pts_right0, pts_left1,
                     K, P1, P2, threshold=SUPPORTER_THRESHOLD):
    """
    Counts how many 3D points project within `threshold` pixels on the valid images.
    Returns: boolean mask of supporters, count of supporters
    """
    P_left0 = P1
    P_right0 = P2
    P_left1 = K @ T_left1

    proj_l0 = project_points(points_3d, P_left0)
    proj_r0 = project_points(points_3d, P_right0)
    proj_l1 = project_points(points_3d, P_left1)

    err_l0 = np.linalg.norm(proj_l0 - pts_left0, axis=1)
    err_r0 = np.linalg.norm(proj_r0 - pts_right0, axis=1)
    err_l1 = np.linalg.norm(proj_l1 - pts_left1, axis=1)

    supporter_mask = (err_l0 < threshold) & (err_r0 < threshold) & (err_l1 < threshold)

    return supporter_mask, int(np.sum(supporter_mask))


def plot_supporters_on_images(frame_idx0, frame_idx1, pts_left0, pts_left1,
                               supporter_mask):
    """Plots matches on left_0 and left_1 with supporters in a different color."""
    left0_img, _ = read_images(frame_idx0)
    left1_img, _ = read_images(frame_idx1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6))

    ax0.imshow(left0_img, cmap='gray')
    ax0.set_title(f"left_{frame_idx0}")
    ax0.axis('off')

    ax1.imshow(left1_img, cmap='gray')
    ax1.set_title(f"left_{frame_idx1}")
    ax1.axis('off')

    outlier_mask = ~supporter_mask

    # Outliers first (cyan)
    ax0.scatter(pts_left0[outlier_mask, 0], pts_left0[outlier_mask, 1],
                c='cyan', s=8, label='Outliers')
    ax1.scatter(pts_left1[outlier_mask, 0], pts_left1[outlier_mask, 1],
                c='cyan', s=8)

    # Supporters on top (orange)
    ax0.scatter(pts_left0[supporter_mask, 0], pts_left0[supporter_mask, 1],
                c='orange', s=8, label='Supporters')
    ax1.scatter(pts_left1[supporter_mask, 0], pts_left1[supporter_mask, 1],
                c='orange', s=8)

    ax0.legend()
    plt.suptitle("Matches on left images – supporters vs outliers")
    plt.tight_layout()
    plt.show()


def section_3_4(pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common,
                T_mat):
    """
    Section 3.4: Count supporters for the T computed in 3.3 and plot them.
    """
    print("--- Section 3.4 ---")

    K, P1, P2 = get_k_and_p_matrices()

    supporter_mask, num_supporters = count_supporters(
        T_mat, pts_3d_common, pts_2d_l0_common, pts_2d_r0_common,
        pts_2d_l1_common, K, P1, P2
    )

    print(f"Number of supporters: {num_supporters} / {len(pts_3d_common)}")

    plot_supporters_on_images(FRAME_0_INDEX, FRAME_1_INDEX,
                               pts_2d_l0_common, pts_2d_l1_common, supporter_mask)

    return supporter_mask


# ==========================================================
# SECTION 3.5 – Dynamic PROSAC + PnP
# ==========================================================
def ransac_pnp(pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common,
               K, P1, P2,
               max_iterations=MAX_RANSAC_ITERATIONS,
               sample_size=PNP_SAMPLE_SIZE,
               threshold=SUPPORTER_THRESHOLD,
               max_points=MAX_RANSAC_POINTS,
               confidence=RANSAC_CONFIDENCE):
    """
    Custom RANSAC with PROSAC sampling and dynamic iteration stopping criterion.
    Returns: best T (3x4), inlier mask
    """
    # 1. Limit points to MAX_RANSAC_POINTS to optimize runtime
    if len(pts_3d_common) > max_points:
        pts_3d_common = pts_3d_common[:max_points]
        pts_2d_l1_common = pts_2d_l1_common[:max_points]
        pts_2d_l0_common = pts_2d_l0_common[:max_points]
        pts_2d_r0_common = pts_2d_r0_common[:max_points]

    n = len(pts_3d_common)
    best_num_supporters = 0
    best_mask = np.zeros(n, dtype=bool)
    best_T = None

    if n < sample_size:
        return best_T, best_mask

    # PROSAC dynamic limits
    # We expand the sample pool (m) iteratively.
    # a_target defines how many samples to draw from a pool size m before expanding.
    a_target = max(1, max_iterations // max(1, n - sample_size + 1))
    m = sample_size - 1
    a_count = 0
    iteration = 0

    num_iterations = max_iterations

    while iteration < num_iterations:
        # PROSAC Selection:
        # Sample the m-th element and 3 other elements from [0, m-1]
        if m < n:
            idx = list(np.random.choice(m, sample_size - 1, replace=False))
            sample_indices = np.array(idx + [m])
            a_count += 1
            if a_count >= a_target:
                m += 1
                a_count = 0
        else:
            # Fallback to standard uniform sampling if we exceed the sorted pool
            sample_indices = np.random.choice(n, sample_size, replace=False)

        sample_3d = pts_3d_common[sample_indices].astype(np.float64)
        sample_2d = pts_2d_l1_common[sample_indices].astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            sample_3d, sample_2d, K, None, flags=cv2.SOLVEPNP_EPNP
        )
        if success:
            T = rodriguez_to_mat(rvec, tvec)
            mask, num_sup = count_supporters(
                T, pts_3d_common, pts_2d_l0_common, pts_2d_r0_common,
                pts_2d_l1_common, K, P1, P2, threshold
            )

            if num_sup > best_num_supporters:
                best_num_supporters = num_sup
                best_mask = mask
                best_T = T

                # Dynamic RANSAC iterations stopping criteria
                w = num_sup / float(n)  # Inlier ratio
                if w < 1.0:
                    denom = math.log(1 - w**sample_size)
                    # To prevent division by zero or errors
                    if denom < -1e-8:
                        i_dynamic = math.log(1 - confidence) / denom
                        num_iterations = min(num_iterations, int(math.ceil(i_dynamic)))
                else:
                    num_iterations = 0  # 100% inliers, stop immediately

        iteration += 1

    # Refine: re-run PnP on ALL identified inliers
    if best_T is not None and np.sum(best_mask) >= sample_size:
        inlier_3d = pts_3d_common[best_mask].astype(np.float64)
        inlier_2d = pts_2d_l1_common[best_mask].astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            inlier_3d, inlier_2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            best_T = rodriguez_to_mat(rvec, tvec)
            # Recount after refinement
            best_mask, best_num_supporters = count_supporters(
                best_T, pts_3d_common, pts_2d_l0_common, pts_2d_r0_common,
                pts_2d_l1_common, K, P1, P2, threshold
            )

    return best_T, best_mask


def section_3_5(point_cloud_0, point_cloud_1,
                pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common):
    """
    Section 3.5: RANSAC + PnP.
    - Plot the two point clouds (pair 0 transformed by T, and pair 1).
    - Plot inliers/outliers on left_0 and left_1.
    """
    print("--- Section 3.5 ---")

    K, P1, P2 = get_k_and_p_matrices()

    best_T, inlier_mask = ransac_pnp(
        pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common,
        K, P1, P2
    )

    print(f"RANSAC inliers: {int(np.sum(inlier_mask))} / {len(inlier_mask)}")
    print(f"Best T:\n{np.round(best_T, 4)}")

    # --- Plot the two point clouds ---
    R = best_T[:3, :3]
    t = best_T[:3, 3]

    # Transform pair 0 cloud into pair 1 coordinates
    cloud0_transformed = (R @ point_cloud_0.points_3d.T).T + t

    # Crop to reasonable depth for display
    cloud0_t = cloud0_transformed
    cloud1 = point_cloud_1.points_3d

    # Simple depth crop: keep points with z in [0, 300]
    mask0 = (cloud0_t[:, 2] > 0) & (cloud0_t[:, 2] < 300)
    mask1 = (cloud1[:, 2] > 0) & (cloud1[:, 2] < 300)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cloud0_t[mask0, 0], cloud0_t[mask0, 1], cloud0_t[mask0, 2],
               s=5, c='tab:blue', alpha=0.4, label='Pair 0 (transformed)')
    ax.scatter(cloud1[mask1, 0], cloud1[mask1, 1], cloud1[mask1, 2],
               s=5, c='tab:orange', alpha=0.4, label='Pair 1')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Aligned Point Clouds (pair 0 after T, pair 1)")
    ax.view_init(elev=-70, azim=-90)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot inliers / outliers on left images ---
    plot_supporters_on_images(FRAME_0_INDEX, FRAME_1_INDEX,
                               pts_2d_l0_common, pts_2d_l1_common, inlier_mask)

    return best_T


# ==========================================================
# SECTION 3.6 – Full sequence tracking
# ==========================================================
def track_pair(frame_idx0, frame_idx1, K, P1, P2, threshold=DEVIATION_THRESHOLD):
    """
    Performs one step of the tracking pipeline between consecutive frames:
    1. Build stereo point cloud for pair 0.
    2. Match left_0 to left_1.
    3. Find common points.
    4. PROSAC-RANSAC + PnP.

    Returns: best T (3x4), point_cloud for frame_idx0
    """
    pc0 = create_stereo_point_cloud(frame_idx0, threshold)

    left0_img, _ = read_images(frame_idx0)
    left1_img, _ = read_images(frame_idx1)

    kp_l0, kp_l1, matches = extract_and_match_features(left0_img, left1_img)

    # Sort matches by distance (ascending) for PROSAC
    matches = sorted(matches, key=lambda m: m.distance)

    l0_pts, l1_pts = get_matched_points(kp_l0, kp_l1, matches)

    pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common = \
        find_common_points(pc0, l0_pts, l1_pts)

    if len(pts_3d_common) < PNP_SAMPLE_SIZE:
        return np.hstack((np.eye(3), np.zeros((3, 1)))), pc0

    best_T, _ = ransac_pnp(
        pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common,
        K, P1, P2
    )

    if best_T is None:
        return np.hstack((np.eye(3), np.zeros((3, 1)))), pc0

    return best_T, pc0


def read_ground_truth_poses(poses_path=POSES_PATH):
    """
    Reads the ground-truth extrinsic matrices from poses file.
    Each line has 12 numbers = the 3x4 [R|t] matrix in row-major order.
    """
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 12:
                poses.append(np.array(values).reshape(3, 4))
    return poses


def section_3_6():
    """
    Section 3.6: Track the full sequence.
    """
    print("--- Section 3.6 ---")

    K, P1, P2 = get_k_and_p_matrices()

    img_dir = SEQ_DIR / "image_0"
    num_frames = len(list(img_dir.glob("*.png")))
    print(f"Total frames in sequence: {num_frames}")

    global_transform = np.eye(4)
    trajectory = [np.array([0.0, 0.0, 0.0])]

    start_time = time.time()

    for i in range(num_frames - 1):
        if i % 100 == 0:
            print(f"  Processing frame {i}/{num_frames - 1} ...")

        T_rel_3x4, _ = track_pair(i, i + 1, K, P1, P2)

        T_rel = np.eye(4)
        T_rel[:3, :] = T_rel_3x4

        global_transform = T_rel @ global_transform

        R_acc = global_transform[:3, :3]
        t_acc = global_transform[:3, 3]
        cam_position = -R_acc.T @ t_acc

        trajectory.append(cam_position)

    elapsed = time.time() - start_time
    print(f"Tracking completed in {elapsed:.1f} seconds.")

    trajectory = np.array(trajectory)

    # --- Read ground truth ---
    gt_poses = read_ground_truth_poses()
    gt_locations = []
    for pose in gt_poses:
        R_gt = pose[:3, :3]
        t_gt = pose[:3, 3]
        C_gt = -R_gt.T @ t_gt
        gt_locations.append(C_gt)
    gt_locations = np.array(gt_locations)

    # --- Plot trajectory ---
    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 2], 'b-', linewidth=1, label='Estimated')
    plt.plot(gt_locations[:num_frames, 0], gt_locations[:num_frames, 2],
             'r-', linewidth=1, label='Ground Truth')
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title("Camera Trajectory – Top-Down View (left_0 coordinates)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    # Section 3.1
    point_cloud_0, point_cloud_1 = section_3_1()

    # Section 3.2
    pts_left0, pts_left1, temporal_matches = section_3_2()

    # Section 3.3
    pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common, T_mat = \
        section_3_3(point_cloud_0, pts_left0, pts_left1)

    # Section 3.4
    supporter_mask = section_3_4(
        pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common, T_mat
    )

    # Section 3.5
    best_T = section_3_5(
        point_cloud_0, point_cloud_1,
        pts_3d_common, pts_2d_l1_common, pts_2d_l0_common, pts_2d_r0_common
    )

    # Section 3.6
    section_3_6()


if __name__ == "__main__":
    main()