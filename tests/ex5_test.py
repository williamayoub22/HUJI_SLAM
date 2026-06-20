import sys

import gtsam
import numpy as np


def flush_print(msg):
    print(msg)
    sys.stdout.flush()


def test_gtsam_basics():
    flush_print("Testing GTSAM basics...")

    # 1. Explicit elements
    flush_print("\n1. Testing Rot3 with explicit elements (R11, R12, etc.)...")
    rot_explicit = gtsam.Rot3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    flush_print("   -> Success")

    # 2. Identity matrix
    flush_print("\n2. Testing Rot3 with identity matrix...")
    R_eye = np.eye(3, dtype=np.float64)
    rot_eye = gtsam.Rot3(R_eye)
    flush_print("   -> Success")

    # 3. Random matrix (SVD orthogonalized)
    flush_print("\n3. Testing Rot3 with SVD-orthogonalized random matrix...")
    R_rand = np.random.rand(3, 3)
    U, _, Vh = np.linalg.svd(R_rand)
    R_ortho = U @ Vh
    if np.linalg.det(R_ortho) < 0:
        U[:, 2] *= -1
        R_ortho = U @ Vh
    rot_ortho = gtsam.Rot3(R_ortho)
    flush_print("   -> Success")

    # 4. From global_camera_matrices
    flush_print("\n4. Testing Rot3 with actual data from ex3...")
    path = "../outputs/ex3/global_camera_matrices.npy"
    mats = np.load(path)
    R = mats[0][:3, :3]
    R_contig = np.ascontiguousarray(R.T)
    rot_data = gtsam.Rot3(R_contig)
    flush_print("   -> Success")


if __name__ == "__main__":
    test_gtsam_basics()
    flush_print("\nAll tests finished successfully!")
