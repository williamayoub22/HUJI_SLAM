import cv2
import numpy as np
import math

# Default hyperparameters (matching ex3_orb.py)
MAX_RANSAC_ITERS = 200
RANSAC_CONFIDENCE = 0.99
SUPPORTER_THRESH = 2.0
MAX_TRANSLATION = 30.0
MIN_INLIERS = 6


def find_common_points(pc0, pc1, l0_pts, l1_pts):
    """
    Finds points existing in both stereo point clouds and temporal matches.

    Args:
        pc0: StereoPointCloud for pair 0
        pc1: StereoPointCloud for pair 1
        l0_pts: Nx2 temporal match points on previous left image (left_0)
        l1_pts: Nx2 temporal match points on current left image (left_1)

    Returns:
        pts_3d: common 3D points from stereo pair 0 triangulation
        pts_l1: corresponding 2D points on left_1
        pts_l0: corresponding 2D points on left_0
        pts_r0: corresponding 2D points on right_0
        pts_r1: corresponding 2D points on right_1
    """
    # Map 2D point locations to their index in the respective point clouds
    stereo0_dict = {(float(pt[0]), float(pt[1])): i for i, pt in enumerate(pc0.left_inliers)}
    stereo1_dict = {(float(pt[0]), float(pt[1])): i for i, pt in enumerate(pc1.left_inliers)}

    idx_temporal, idx_stereo0, idx_stereo1 = [], [], []
    for i, (pt0, pt1) in enumerate(zip(l0_pts, l1_pts)):
        key0 = (float(pt0[0]), float(pt0[1]))
        key1 = (float(pt1[0]), float(pt1[1]))

        # Keep only key-points matched on all four images
        if key0 in stereo0_dict and key1 in stereo1_dict:
            idx_temporal.append(i)
            idx_stereo0.append(stereo0_dict[key0])
            idx_stereo1.append(stereo1_dict[key1])

    return (pc0.points_3d[idx_stereo0],
            l1_pts[idx_temporal],
            l0_pts[idx_temporal],
            pc0.right_inliers[idx_stereo0],
            pc1.right_inliers[idx_stereo1])


def project_points(pts_3d, P):
    """Projects Nx3 3D points using a 3x4 projection matrix. Returns Nx2."""
    pts_h = np.hstack((pts_3d, np.ones((len(pts_3d), 1))))
    proj = (P @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


def count_supporters(T, pts_3d, pts_l0, pts_r0, pts_l1, pts_r1, K, P1, P2,
                     threshold=SUPPORTER_THRESH):
    """Counts points projecting within threshold pixels on all four images."""
    P_l1 = K @ T

    # T transforms from left_0 to left_1. P2 projects from left to right.
    # Therefore, the right_1 projection matrix is P2 @ T_4x4
    T_4x4 = np.vstack((T, [0, 0, 0, 1]))
    P_r1 = P2 @ T_4x4

    err_l0 = np.linalg.norm(project_points(pts_3d, P1) - pts_l0, axis=1)
    err_r0 = np.linalg.norm(project_points(pts_3d, P2) - pts_r0, axis=1)
    err_l1 = np.linalg.norm(project_points(pts_3d, P_l1) - pts_l1, axis=1)
    err_r1 = np.linalg.norm(project_points(pts_3d, P_r1) - pts_r1, axis=1)

    mask = (err_l0 < threshold) & (err_r0 < threshold) & (err_l1 < threshold) & (err_r1 < threshold)
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


def ransac_pnp(pts_3d, pts_l1, pts_l0, pts_r0, pts_r1, K, P1, P2,
               max_iters=MAX_RANSAC_ITERS,
               confidence=RANSAC_CONFIDENCE,
               supporter_thresh=SUPPORTER_THRESH,
               max_translation=MAX_TRANSLATION,
               min_inliers=MIN_INLIERS):
    """
    RANSAC + PnP with dynamic iteration cap, translation sanity check,
    and minimum inlier gate.

    Returns: (best_T, best_mask) or (None, None) if failed.
    """
    best_mask, best_T, max_supporters = None, None, -1
    n = len(pts_3d)

    if n < 4:
        return None, None

    current_max_iters = max_iters
    iterations = 0

    while iterations < current_max_iters:
        idx = np.random.choice(n, 4, replace=False)
        T = _safe_solvePnP(pts_3d[idx], pts_l1[idx], K, cv2.SOLVEPNP_EPNP)

        if T is not None:
            t_norm = np.linalg.norm(T[:3, 3])
            if t_norm < max_translation:
                mask, count = count_supporters(
                    T, pts_3d, pts_l0, pts_r0, pts_l1, pts_r1, K, P1, P2, supporter_thresh
                )

                if count > max_supporters:
                    max_supporters, best_mask, best_T = count, mask, T

                    w = count / n
                    if w > 0:
                        prob_good_sample = w ** 4
                        prob_bad_sample = max(1e-10, min(1.0 - 1e-10, 1.0 - prob_good_sample))

                        try:
                            dynamic_iters = math.log(1.0 - confidence) / math.log(prob_bad_sample)
                            current_max_iters = min(current_max_iters, int(math.ceil(dynamic_iters)))
                        except ZeroDivisionError:
                            current_max_iters = 0

        iterations += 1

    # Refine on all inliers
    if max_supporters >= min_inliers:
        T_refined = _safe_solvePnP(pts_3d[best_mask], pts_l1[best_mask], K, cv2.SOLVEPNP_ITERATIVE)
        if T_refined is not None and np.linalg.norm(T_refined[:3, 3]) < max_translation:
            best_T = T_refined
            best_mask, max_supporters = count_supporters(
                best_T, pts_3d, pts_l0, pts_r0, pts_l1, pts_r1, K, P1, P2, supporter_thresh
            )

    if max_supporters < min_inliers:
        return None, None

    return best_T, best_mask