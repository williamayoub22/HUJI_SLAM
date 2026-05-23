import numpy as np

SUPPORTER_THRESH = 2.0

def project_points(pts_3d, P):
    """Projects Nx3 3D points using a 3x4 projection matrix. Returns Nx2."""
    pts_h = np.hstack((pts_3d, np.ones((len(pts_3d), 1))))
    proj = (P @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


def count_supporters(T, pts_3d, pts_l0, pts_r0, pts_l1, pts_r1, K, P1, P2,
                     threshold=SUPPORTER_THRESH):
    """Counts points projecting within threshold pixels on all four images."""
    P_l1 = K @ T
    T_4x4 = np.vstack((T, [0, 0, 0, 1]))
    P_r1 = P2 @ T_4x4

    err_l0 = np.linalg.norm(project_points(pts_3d, P1) - pts_l0, axis=1)
    err_r0 = np.linalg.norm(project_points(pts_3d, P2) - pts_r0, axis=1)
    err_l1 = np.linalg.norm(project_points(pts_3d, P_l1) - pts_l1, axis=1)
    err_r1 = np.linalg.norm(project_points(pts_3d, P_r1) - pts_r1, axis=1)

    mask = (err_l0 < threshold) & (err_r0 < threshold) & (err_l1 < threshold) & (err_r1 < threshold)
    return mask, np.sum(mask)