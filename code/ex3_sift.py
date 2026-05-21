import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from pathlib import Path

# Assume these utils are provided by your assignment framework
from utils.image_loader import read_images
from utils.matching import get_matched_points
from utils.visualization import plot_point_cloud_on_axis
from utils.stereo_pipeline import create_stereo_point_cloud
from utils.read_cam_calib import read_calib

# --- Hyperparameters ---
FRAME_0, FRAME_1 = 0, 1
MAX_RANSAC_ITERS = 50  # Absolute cap on iterations
RANSAC_CONFIDENCE = 0.99  # 'p' probability of success
SUPPORTER_THRESH = 2.0  # pixels
RATIO_THRESHOLD = 0.7  # Lowe's Ratio Test threshold
NUM_FEATURES = 2000  # SIFT feature limit

SEQ_DIR = Path(__file__).resolve().parent.parent / "dataset" / "sequences" / "00"
POSES_PATH = Path(__file__).resolve().parent.parent / "dataset" / "poses" / "00.txt"


def get_calib():
    P1, P2 = read_calib()
    return P1[:, :3], P1, P2


def find_common_points(pc0, l0_pts, l1_pts):
    """Finds points existing in both stereo pair 0 and temporal matches."""
    stereo_dict = {(float(pt[0]), float(pt[1])): i for i, pt in enumerate(pc0.left_inliers)}

    idx_temporal, idx_stereo = [], []
    for i, pt in enumerate(l0_pts):
        key = (float(pt[0]), float(pt[1]))
        if key in stereo_dict:
            idx_temporal.append(i)
            idx_stereo.append(stereo_dict[key])

    return (pc0.points_3d[idx_stereo], l1_pts[idx_temporal],
            l0_pts[idx_temporal], pc0.right_inliers[idx_stereo])


def project_points(pts_3d, P):
    """Projects 3D points using 3x4 projection matrix."""
    pts_h = np.hstack((pts_3d, np.ones((len(pts_3d), 1))))
    proj = (P @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


def count_supporters(T, pts_3d, pts_l0, pts_r0, pts_l1, K, P1, P2):
    """Counts points projecting within SUPPORTER_THRESH on all valid images."""
    P_l1 = K @ T
    err_l0 = np.linalg.norm(project_points(pts_3d, P1) - pts_l0, axis=1)
    err_r0 = np.linalg.norm(project_points(pts_3d, P2) - pts_r0, axis=1)
    err_l1 = np.linalg.norm(project_points(pts_3d, P_l1) - pts_l1, axis=1)

    mask = (err_l0 < SUPPORTER_THRESH) & (err_r0 < SUPPORTER_THRESH) & (err_l1 < SUPPORTER_THRESH)
    return mask, np.sum(mask)


def ransac_pnp(pts_3d, pts_l1, pts_l0, pts_r0, K, P1, P2):
    """RANSAC implementation with dynamic iteration cap."""
    best_mask, best_T, max_supporters = None, None, -1
    n = len(pts_3d)

    if n < 4: return None, None

    max_iters = MAX_RANSAC_ITERS
    iterations = 0

    # Dynamic RANSAC loop
    while iterations < max_iters:
        idx = np.random.choice(n, 4, replace=False)
        success, rvec, tvec = cv2.solvePnP(pts_3d[idx], pts_l1[idx], K, None, flags=cv2.SOLVEPNP_EPNP)

        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.hstack((R, tvec))

            mask, count = count_supporters(T, pts_3d, pts_l0, pts_r0, pts_l1, K, P1, P2)

            if count > max_supporters:
                max_supporters, best_mask, best_T = count, mask, T

                # --- Update dynamic iterations ---
                w = count / n
                if w > 0:
                    prob_good_sample = w ** 4
                    prob_bad_sample = max(1e-10, min(1.0 - 1e-10, 1.0 - prob_good_sample))

                    try:
                        dynamic_iters = math.log(1.0 - RANSAC_CONFIDENCE) / math.log(prob_bad_sample)
                        max_iters = min(max_iters, int(math.ceil(dynamic_iters)))
                    except ZeroDivisionError:
                        max_iters = 0

        iterations += 1

    # Refine on all inliers
    if max_supporters >= 4:
        success, rvec, tvec = cv2.solvePnP(pts_3d[best_mask], pts_l1[best_mask], K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            best_T = np.hstack((R, tvec))
            best_mask, _ = count_supporters(best_T, pts_3d, pts_l0, pts_r0, pts_l1, K, P1, P2)

    return best_T, best_mask


def track_sequence():
    """Runs the full tracking pipeline with Caching & Lowe's Ratio Test."""
    K, P1, P2 = get_calib()
    num_frames = len(list((SEQ_DIR / "image_0").glob("*.png")))

    global_T = np.eye(4)
    trajectory = [np.zeros(3)]

    t0 = time.time()

    # --- CACHING SETUP (Optimization) ---
    # Setup SIFT and Matcher once
    sift = cv2.SIFT_create(nfeatures=NUM_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Initialize the "previous" state using Frame 0 before the loop begins
    img_prev, _ = read_images(0)
    kp_prev, des_prev = sift.detectAndCompute(img_prev, None)
    pc_prev = create_stereo_point_cloud(0)

    for i in range(num_frames - 1):
        if i % 50 == 0: print(f"Processing frame {i}/{num_frames - 1}...")

        # 1. Load and compute ONLY the "current" frame (i+1)
        img_curr, _ = read_images(i + 1)
        kp_curr, des_curr = sift.detectAndCompute(img_curr, None)

        # 2. Extract and Match Features with k=2
        raw_matches = bf.knnMatch(des_prev, des_curr, k=2)

        # 3. Apply Lowe's Ratio Test
        good_matches = []
        for m, n in raw_matches:
            if m.distance < RATIO_THRESHOLD * n.distance:
                good_matches.append(m)

        # Extract 2D points from the filtered matches
        pts_l0, pts_l1 = get_matched_points(kp_prev, kp_curr, good_matches)

        # Find 3D points and run PnP
        pts_3d, pts_l1_c, pts_l0_c, pts_r0_c = find_common_points(pc_prev, pts_l0, pts_l1)
        T_rel, _ = ransac_pnp(pts_3d, pts_l1_c, pts_l0_c, pts_r0_c, K, P1, P2)

        if T_rel is None:
            T_rel = np.hstack((np.eye(3), np.zeros((3, 1))))  # Fallback if RANSAC fails

        step_T = np.eye(4)
        step_T[:3, :] = T_rel
        global_T = step_T @ global_T

        # Calculate position: -R^T * t
        pos = -global_T[:3, :3].T @ global_T[:3, 3]
        trajectory.append(pos)

        # --- SHIFT THE CACHE ---
        # The 'current' frame data becomes the 'previous' frame data for the next loop
        img_prev = img_curr
        kp_prev = kp_curr
        des_prev = des_curr
        pc_prev = create_stereo_point_cloud(i + 1)

    print(f"Tracking took {time.time() - t0:.2f}s")
    return np.array(trajectory)


def plot_trajectory(trajectory):
    """Plots estimated vs ground truth trajectory."""
    gt_poses = [np.array([float(x) for x in line.split()]).reshape(3, 4)
                for line in open(POSES_PATH)]
    gt_traj = np.array([-p[:3, :3].T @ p[:3, 3] for p in gt_poses])

    plt.figure(figsize=(10, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 2], 'b-', label='Estimated')
    plt.plot(gt_traj[:len(trajectory), 0], gt_traj[:len(trajectory), 2], 'r-', label='Ground Truth')
    plt.title("Camera Trajectory (X-Z plane)")
    plt.xlabel("X (m)");
    plt.ylabel("Z (m)")
    plt.legend();
    plt.axis('equal');
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Start the total execution timer
    start_time = time.time()

    # Run the visual odometry pipeline
    traj = track_sequence()

    # Plot the results
    plot_trajectory(traj)

    # Calculate and print total time
    elapsed_seconds = time.time() - start_time
    elapsed_minutes = int(elapsed_seconds // 60)
    remaining_seconds = elapsed_seconds % 60

    print(f"Total execution time: {elapsed_minutes} minutes and {remaining_seconds:.2f} seconds.")