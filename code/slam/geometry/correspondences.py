import numpy as np

from ..pipeline.stereo_pipeline import StereoPointCloud


def find_common_points(
    pc0: StereoPointCloud,
    pc1: StereoPointCloud,
    l0_pts: np.ndarray,
    l1_pts: np.ndarray,
    tolerance: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds points observed in all four images: left_0, right_0, left_1, right_1.

    pc0 gives the 3D point from stereo pair 0, together with its left_0/right_0 pixels.
    The temporal matches give left_0 <-> left_1.
    pc1 verifies that the left_1 point also has a stereo match to right_1.
    """
    pts_3d = []
    pts_l0 = []
    pts_r0 = []
    pts_l1 = []
    pts_r1 = []

    for temporal_idx, l0_pt in enumerate(l0_pts):
        l1_pt = l1_pts[temporal_idx]

        dist0 = np.linalg.norm(pc0.left_inliers - l0_pt, axis=1)
        idx0 = int(np.argmin(dist0))

        dist1 = np.linalg.norm(pc1.left_inliers - l1_pt, axis=1)
        idx1 = int(np.argmin(dist1))

        if dist0[idx0] <= tolerance and dist1[idx1] <= tolerance:
            pts_3d.append(pc0.points_3d[idx0])
            pts_l0.append(pc0.left_inliers[idx0])
            pts_r0.append(pc0.right_inliers[idx0])
            pts_l1.append(pc1.left_inliers[idx1])
            pts_r1.append(pc1.right_inliers[idx1])

    return (
        np.asarray(pts_3d, dtype=np.float64),
        np.asarray(pts_l1, dtype=np.float64),
        np.asarray(pts_l0, dtype=np.float64),
        np.asarray(pts_r0, dtype=np.float64),
        np.asarray(pts_r1, dtype=np.float64),
    )
