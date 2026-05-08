import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import List, Tuple

# Optional: Ensure 3D plotting backend is loaded
from mpl_toolkits.mplot3d import Axes3D

# Custom imports based on your project structure
from utils.features import extract_features
from utils.matching import match_features
from utils.image_loader import read_images
from utils.read_cam_calib import read_calib

# ==========================================
# CONFIGURATION
# ==========================================
FRAME_INDEX = 0
VERTICAL_DEVIATION_THRESHOLD = 2.0


# ==========================================
# GEOMETRY HELPERS
# ==========================================
def get_matched_points(
        kp_left: List[cv2.KeyPoint],
        kp_right: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns matched pixel locations as two Nx2 arrays:
    left_pts[i] = (x_left, y_left)
    right_pts[i] = (x_right, y_right)
    """
    left_pts = np.array([kp_left[m.queryIdx].pt for m in matches])
    right_pts = np.array([kp_right[m.trainIdx].pt for m in matches])

    return left_pts, right_pts


def compute_vertical_deviations(
        kp_left: List[cv2.KeyPoint],
        kp_right: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
) -> np.ndarray:
    left_pts, right_pts = get_matched_points(kp_left, kp_right, matches)

    y_left = left_pts[:, 1]
    y_right = right_pts[:, 1]

    deviations = np.abs(y_left - y_right)
    return deviations


def plot_vertical_deviation_histogram(deviations: np.ndarray):
    plt.figure(figsize=(8, 5))
    plt.hist(deviations, bins=50)
    plt.xlabel(r"Vertical deviation $|y_L - y_R|$ [pixels]")
    plt.ylabel("Number of matches")
    plt.title("Histogram of vertical deviations from rectified stereo pattern")
    plt.grid(True)
    plt.show()


def triangulate_dlt_custom(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
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

        # Build the 4x4 A matrix
        A = np.zeros((4, 4))
        A[0] = x1 * P1[2, :] - P1[0, :]
        A[1] = y1 * P1[2, :] - P1[1, :]
        A[2] = x2 * P2[2, :] - P2[0, :]
        A[3] = y2 * P2[2, :] - P2[1, :]

        # Solve AX = 0 using SVD
        U, S, Vt = np.linalg.svd(A)

        # The solution is the last column of V, which is the last row of Vt
        X_homogeneous = Vt[-1, :]

        # Convert from homogeneous (x,y,z,w) to Cartesian (x/w, y/w, z/w)
        points_3d[i] = X_homogeneous[:3] / X_homogeneous[3]

    return points_3d


def plot_3d_point_cloud(points_3d: np.ndarray, title: str):
    """Plots a 3D scatter of the point cloud."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    xs = points_3d[:, 0]
    ys = points_3d[:, 1]
    zs = points_3d[:, 2]

    ax.scatter(xs, ys, zs, s=10, c='tab:blue', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Adjust viewing angle to match typical stereo point cloud rendering
    ax.view_init(elev=-70, azim=-90)
    plt.show()


# ==========================================
# ASSIGNMENT SECTIONS
# ==========================================
def section_2_1(frame_idx: int = FRAME_INDEX):
    left_img, right_img = read_images(frame_idx)

    kp_left, des_left = extract_features(left_img)
    kp_right, des_right = extract_features(right_img)

    knn_matches = match_features(des_left, des_right)
    matches = [m[0] for m in knn_matches if len(m) > 0]

    deviations = compute_vertical_deviations(kp_left, kp_right, matches)

    plot_vertical_deviation_histogram(deviations)

    num_matches = len(matches)
    num_bad = np.sum(deviations > VERTICAL_DEVIATION_THRESHOLD)
    percentage_bad = 100.0 * num_bad / num_matches

    print(f"--- Section 2.1 ---")
    print(f"Total matches: {num_matches}")
    print(f"Matches with vertical deviation > {VERTICAL_DEVIATION_THRESHOLD} px: {num_bad}")
    print(f"Percentage: {percentage_bad:.2f}%\n")

    return {
        "left_img": left_img,
        "right_img": right_img,
        "kp_left": kp_left,
        "kp_right": kp_right,
        "matches": matches,
        "deviations": deviations,
    }


def section_2_2(data_dict: dict, threshold: float = VERTICAL_DEVIATION_THRESHOLD):
    left_img = data_dict["left_img"]
    right_img = data_dict["right_img"]
    kp_left = data_dict["kp_left"]
    kp_right = data_dict["kp_right"]
    matches = data_dict["matches"]
    deviations = data_dict["deviations"]

    # 1. Get matched points
    left_pts, right_pts = get_matched_points(kp_left, kp_right, matches)

    # 2. Create boolean masks
    inlier_mask = deviations <= threshold
    outlier_mask = deviations > threshold

    # 3. Calculate discarded matches
    num_discarded = np.sum(outlier_mask)
    num_total = len(matches)

    print(f"--- Section 2.2 ---")
    print(f"Total matches: {num_total}")
    print(f"Matches discarded (outliers): {num_discarded}\n")

    # 4. Plotting (Side-by-side without connecting lines)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cmap_left = 'gray' if len(left_img.shape) == 2 else None
    cmap_right = 'gray' if len(right_img.shape) == 2 else None

    ax1.imshow(left_img, cmap=cmap_left)
    ax1.set_title("Left Image")
    ax1.axis('off')

    ax2.imshow(right_img, cmap=cmap_right)
    ax2.set_title("Right Image")
    ax2.axis('off')

    # Plot outliers FIRST (Cyan) so they are at the bottom
    ax1.scatter(left_pts[outlier_mask, 0], left_pts[outlier_mask, 1], c='cyan', s=15, label='Rejected (Outliers)')
    ax2.scatter(right_pts[outlier_mask, 0], right_pts[outlier_mask, 1], c='cyan', s=15)

    # Plot inliers SECOND (Orange) so they appear on top
    ax1.scatter(left_pts[inlier_mask, 0], left_pts[inlier_mask, 1], c='orange', s=15, label='Accepted (Inliers)')
    ax2.scatter(right_pts[inlier_mask, 0], right_pts[inlier_mask, 1], c='orange', s=15)

    fig.legend(loc='lower center', ncol=2)
    plt.suptitle(f"Rectified Stereo Match Rejection (Threshold: {threshold} px)")
    plt.tight_layout()
    plt.show()

    return num_discarded


def section_2_3(data_dict: dict, threshold: float = VERTICAL_DEVIATION_THRESHOLD):
    print(f"--- Section 2.3 ---")

    # 1. Read camera matrices (automatically finds the dataset path)
    P1, P2 = read_calib()

    # 2. Extract inlier matches
    kp_left = data_dict["kp_left"]
    kp_right = data_dict["kp_right"]
    matches = data_dict["matches"]
    deviations = data_dict["deviations"]

    left_pts, right_pts = get_matched_points(kp_left, kp_right, matches)
    inlier_mask = deviations <= threshold

    pts1_inliers = left_pts[inlier_mask]
    pts2_inliers = right_pts[inlier_mask]

    # 3. Custom Triangulation (DLT)
    custom_3d_points = triangulate_dlt_custom(P1, P2, pts1_inliers, pts2_inliers)

    # 4. OpenCV Triangulation
    cv_4d_points = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
    cv_3d_points = (cv_4d_points[:3, :] / cv_4d_points[3, :]).T

    # 5. Plot Side-by-Side on a single figure
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(custom_3d_points[:, 0], custom_3d_points[:, 1], custom_3d_points[:, 2], s=10, c='tab:blue', alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title("Custom DLT Triangulation")
    ax1.view_init(elev=-70, azim=-90)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(cv_3d_points[:, 0], cv_3d_points[:, 1], cv_3d_points[:, 2], s=10, c='tab:orange', alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title("OpenCV Triangulation")
    ax2.view_init(elev=-70, azim=-90)

    plt.tight_layout()
    plt.show()

    # 6. Compare Results
    distances = np.linalg.norm(custom_3d_points - cv_3d_points, axis=1)
    median_distance = np.median(distances)

    print(f"Number of triangulated points: {len(pts1_inliers)}")
    print(f"Median distance between Custom and OpenCV 3D points: {median_distance:.4e}\n")


def section_2_4(frames_to_test: List[int] = [0, 1, 2, 3], threshold: float = VERTICAL_DEVIATION_THRESHOLD):
    print(f"--- Section 2.4: Processing Multiple Frames ---")

    P1, P2 = read_calib()

    # Create a single figure sized for a 2x2 grid
    fig = plt.figure(figsize=(16, 14))

    for idx, frame_idx in enumerate(frames_to_test[:4]):  # Ensure max 4 frames
        print(f"\nProcessing Frame {frame_idx}...")

        left_img, right_img = read_images(frame_idx)
        kp_left, des_left = extract_features(left_img)
        kp_right, des_right = extract_features(right_img)

        knn_matches = match_features(des_left, des_right)
        matches = [m[0] for m in knn_matches if len(m) > 0]

        left_pts, right_pts = get_matched_points(kp_left, kp_right, matches)
        deviations = np.abs(left_pts[:, 1] - right_pts[:, 1])
        inlier_mask = deviations <= threshold

        pts1_inliers = left_pts[inlier_mask]
        pts2_inliers = right_pts[inlier_mask]

        cv_4d_points = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
        points_3d = (cv_4d_points[:3, :] / cv_4d_points[3, :]).T

        print(f"Number of triangulated points: {len(points_3d)}")

        # Add a 3D subplot for this frame, arranged in a 2x2 grid
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        xs = points_3d[:, 0]
        ys = points_3d[:, 1]
        zs = points_3d[:, 2]

        ax.scatter(xs, ys, zs, s=10, c='tab:blue', alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Triangulation Point Cloud (Frame {frame_idx})")
        ax.view_init(elev=-70, azim=-90)

    # Automatically adjust subplot parameters to remove extra padding space
    plt.tight_layout()
    plt.show()


def main():
    data_dict = section_2_1()
    section_2_2(data_dict)
    section_2_3(data_dict)
    section_2_4([0, 1, 2, 3])


if __name__ == "__main__":
    main()