import cv2
import numpy as np


def solve_pnp_safe(pts_3d, pts_2d, K, flags):
    """
    Estimate camera transformation [R | t] from 3D-2D correspondences.
    Returns:
        T: 3x4 matrix [R | t], or None if PnP fails.
    """
    try:
        success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, None, flags=flags)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.hstack((R, tvec))
            return T
    except cv2.error:
        pass
    return None
