import cv2
import math
import numpy as np

from .pnp import solve_pnp_safe
from .projection import count_supporters

MAX_RANSAC_ITERS = 200
RANSAC_CONFIDENCE = 0.99
MAX_TRANSLATION = 30.0
SUPPORTER_THRESH = 2.0
MIN_INLIERS = 6


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
        T = solve_pnp_safe(pts_3d[idx], pts_l1[idx], K, cv2.SOLVEPNP_EPNP)

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
        T_refined = solve_pnp_safe(pts_3d[best_mask], pts_l1[best_mask], K, cv2.SOLVEPNP_ITERATIVE)
        if T_refined is not None and np.linalg.norm(T_refined[:3, 3]) < max_translation:
            best_T = T_refined
            best_mask, max_supporters = count_supporters(
                best_T, pts_3d, pts_l0, pts_r0, pts_l1, pts_r1, K, P1, P2, supporter_thresh
            )

    if max_supporters < min_inliers:
        return None, None

    return best_T, best_mask