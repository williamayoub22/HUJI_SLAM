import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from pathlib import Path

from utils.image_loader import read_images
from utils.matching import get_matched_points
from utils.read_cam_calib import read_calib

# --- Hyperparameters ---
FRAME_0, FRAME_1 = 0, 1
MAX_RANSAC_ITERS = 200       # Dynamic cap converges fast
RANSAC_CONFIDENCE = 0.99     # 'p' probability of success
SUPPORTER_THRESH = 2.0       # pixels
RATIO_THRESHOLD = 0.75       # Lowe's Ratio Test (slightly relaxed for ORB)
NUM_FEATURES = 3000           # ORB needs more features than SIFT to compensate
DEVIATION_THRESHOLD = 2.0    # vertical deviation threshold for stereo outliers
MAX_DEPTH = 300.0            # reject triangulated points farther than this (meters)
MAX_TRANSLATION = 30.0       # reject PnP results with ||t|| > this (meters per frame)
MIN_INLIERS = 6              # minimum RANSAC supporters to accept a result

SEQ_DIR = Path(__file__).resolve().parent.parent / "dataset" / "sequences" / "00"
POSES_PATH = Path(__file__).resolve().parent.parent / "dataset" / "poses" / "00.txt"


def get_calib():
    P1, P2 = read_calib()
    return P1[:, :3], P1, P2


# ─────────────────────────────────────────────
#  Inline ORB-based stereo pipeline
#  (avoids create_stereo_point_cloud which uses SIFT internally)
# ─────────────────────────────────────────────

class StereoResult:
    """Lightweight container matching the fields ex3_orb needs."""
    __slots__ = ['left_inliers', 'right_inliers', 'points_3d']

    def __init__(self, left_inliers, right_inliers, points_3d):
        self.left_inliers = left_inliers
        self.right_inliers = right_inliers
        self.points_3d = points_3d


def orb_stereo_point_cloud(orb, bf, frame_idx, P1, P2):
    """
    Builds a stereo point cloud for *frame_idx* using the supplied ORB
    detector — so keypoints are consistent with temporal matching.
    """
    left_img, right_img = read_images(frame_idx)

    kp_l, des_l = orb.detectAndCompute(left_img, None)
    kp_r, des_r = orb.detectAndCompute(right_img, None)

    empty = StereoResult(np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 3)))

    if des_l is None or des_r is None:
        return empty, kp_l, des_l

    raw = bf.knnMatch(des_l, des_r, k=2)

    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_THRESHOLD * n.distance:
                good.append(m)

    if len(good) == 0:
        return empty, kp_l, des_l

    left_pts, right_pts = get_matched_points(kp_l, kp_r, good)

    # Reject stereo outliers by vertical deviation (rectified images)
    dev = np.abs(left_pts[:, 1] - right_pts[:, 1])
    mask = dev <= DEVIATION_THRESHOLD
    left_pts = left_pts[mask]
    right_pts = right_pts[mask]

    if len(left_pts) == 0:
        return empty, kp_l, des_l

    # Triangulate (OpenCV)
    pts4 = cv2.triangulatePoints(P1, P2, left_pts.T, right_pts.T)
    pts3d = (pts4[:3, :] / pts4[3, :]).T

    # Keep only positive depth AND within MAX_DEPTH
    valid = (pts3d[:, 2] > 0) & (pts3d[:, 2] < MAX_DEPTH)
    left_pts = left_pts[valid]
    right_pts = right_pts[valid]
    pts3d = pts3d[valid]

    return StereoResult(left_pts, right_pts, pts3d), kp_l, des_l


# ─────────────────────────────────────────────
#  Core geometry helpers
# ─────────────────────────────────────────────

def find_common_points(pc, l0_pts, l1_pts):
    """Finds points existing in both stereo pair 0 and temporal matches."""
    stereo_dict = {(float(pt[0]), float(pt[1])): i for i, pt in enumerate(pc.left_inliers)}

    idx_temporal, idx_stereo = [], []
    for i, pt in enumerate(l0_pts):
        key = (float(pt[0]), float(pt[1]))
        if key in stereo_dict:
            idx_temporal.append(i)
            idx_stereo.append(stereo_dict[key])

    return (pc.points_3d[idx_stereo], l1_pts[idx_temporal],
            l0_pts[idx_temporal], pc.right_inliers[idx_stereo])


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


def _safe_solvePnP(pts_3d, pts_2d, K, flags):
    """Wrapper around solvePnP that catches degenerate configurations."""
    try:
        success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, None, flags=flags)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.hstack((R, tvec))
            return T
    except cv2.error:
        pass
    return None


def ransac_pnp(pts_3d, pts_l1, pts_l0, pts_r0, K, P1, P2):
    """RANSAC implementation with dynamic iteration cap."""
    best_mask, best_T, max_supporters = None, None, -1
    n = len(pts_3d)

    if n < 4:
        return None, None

    max_iters = MAX_RANSAC_ITERS
    iterations = 0

    # Dynamic RANSAC loop
    while iterations < max_iters:
        idx = np.random.choice(n, 4, replace=False)
        T = _safe_solvePnP(pts_3d[idx], pts_l1[idx], K, cv2.SOLVEPNP_EPNP)

        if T is not None:
            # Reject obviously bad solutions (huge translation)
            t_norm = np.linalg.norm(T[:3, 3])
            if t_norm < MAX_TRANSLATION:
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
    if max_supporters >= MIN_INLIERS:
        T_refined = _safe_solvePnP(pts_3d[best_mask], pts_l1[best_mask], K, cv2.SOLVEPNP_ITERATIVE)
        if T_refined is not None and np.linalg.norm(T_refined[:3, 3]) < MAX_TRANSLATION:
            best_T = T_refined
            best_mask, max_supporters = count_supporters(best_T, pts_3d, pts_l0, pts_r0, pts_l1, K, P1, P2)

    # Final gate: reject results with too few supporters
    if max_supporters < MIN_INLIERS:
        return None, None

    return best_T, best_mask


# ─────────────────────────────────────────────
#  Main tracking loop
# ─────────────────────────────────────────────

def track_sequence():
    """Runs the full tracking pipeline with Caching, ORB, & Lowe's Ratio Test."""
    K, P1, P2 = get_calib()
    num_frames = len(list((SEQ_DIR / "image_0").glob("*.png")))

    global_T = np.eye(4)
    trajectory = [np.zeros(3)]

    t0 = time.time()

    # Setup ORB and Matcher once (NORM_HAMMING for binary descriptors)
    orb = cv2.ORB_create(nfeatures=NUM_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Initialize "previous" state using Frame 0 — using ORB for BOTH stereo and temporal
    pc_prev, kp_prev, des_prev = orb_stereo_point_cloud(orb, bf, 0, P1, P2)

    for i in range(num_frames - 1):
        if i % 50 == 0:
            print(f"Processing frame {i}/{num_frames - 1}...")

        # 1. Load and compute ONLY the "current" frame (i+1) — left image for temporal matching
        img_curr, _ = read_images(i + 1)
        kp_curr, des_curr = orb.detectAndCompute(img_curr, None)

        # Safety check
        if des_prev is None or des_curr is None:
            pc_prev, kp_prev, des_prev = orb_stereo_point_cloud(orb, bf, i + 1, P1, P2)
            continue

        # 2. Match temporal features (prev left → curr left) with k=2
        raw_matches = bf.knnMatch(des_prev, des_curr, k=2)

        # 3. Apply Lowe's Ratio Test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < RATIO_THRESHOLD * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            pc_prev, kp_prev, des_prev = orb_stereo_point_cloud(orb, bf, i + 1, P1, P2)
            trajectory.append(trajectory[-1].copy())
            continue

        # Extract 2D points from the filtered matches
        pts_l0, pts_l1 = get_matched_points(kp_prev, kp_curr, good_matches)

        # Find 3D points and run PnP
        pts_3d, pts_l1_c, pts_l0_c, pts_r0_c = find_common_points(pc_prev, pts_l0, pts_l1)
        T_rel, _ = ransac_pnp(pts_3d, pts_l1_c, pts_l0_c, pts_r0_c, K, P1, P2)

        if T_rel is None:
            T_rel = np.hstack((np.eye(3), np.zeros((3, 1))))  # Fallback: no motion

        step_T = np.eye(4)
        step_T[:3, :] = T_rel
        global_T = step_T @ global_T

        # Calculate position: -R^T * t
        pos = -global_T[:3, :3].T @ global_T[:3, 3]
        trajectory.append(pos)

        # --- SHIFT THE CACHE ---
        pc_prev, kp_prev, des_prev = orb_stereo_point_cloud(orb, bf, i + 1, P1, P2)

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