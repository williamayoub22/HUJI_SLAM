
import numpy as np

def custom_triangulation(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Triangulates 3D points using the DLT linear least squares method.
    pts1, pts2: Nx2 arrays of 2D coordinates.
    Returns: Nx3 array of 3D points.
    """
    num_points = pts1.shape[0]
    points_3d = np.zeros((num_points, 3))

    for i in range(num_points):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.zeros((4, 4))
        A[0] = x1 * P1[2, :] - P1[0, :]
        A[1] = y1 * P1[2, :] - P1[1, :]
        A[2] = x2 * P2[2, :] - P2[0, :]
        A[3] = y2 * P2[2, :] - P2[1, :]

        U, S, Vt = np.linalg.svd(A)

        X_homogeneous = Vt[-1, :]
        points_3d[i] = X_homogeneous[:3] / X_homogeneous[3]

    return points_3d