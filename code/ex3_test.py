import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from utils.image_loader import read_images
from utils.matching import extract_and_match_features, get_matched_points
from utils.visualization import plot_point_cloud_on_axis
from utils.stereo_pipeline import StereoMatchData, StereoPointCloud, DEVIATION_THRESHOLD
from utils.stereo_pipeline import create_stereo_point_cloud
from utils.read_cam_calib import read_calib

FRAME_0_INDEX = 0
FRAME_1_INDEX = 1
NUM_TEMPORAL_MATCHES_TO_DRAW = 200


# ==========================================
# SECTION 3.1 HELPERS
# ==========================================
# (Relies purely on imports from your utils module)

# ==========================================
# SECTION 3.1
# ==========================================
def section_3_1(
        threshold: float = DEVIATION_THRESHOLD,
) -> tuple[StereoPointCloud, StereoPointCloud]:
    """Creates point clouds for stereo pair 0 and stereo pair 1."""
    print("--- Section 3.1 ---")

    point_cloud_0 = create_stereo_point_cloud(FRAME_0_INDEX, threshold)
    point_cloud_1 = create_stereo_point_cloud(FRAME_1_INDEX, threshold)

    fig = plt.figure(figsize=(16, 7))

    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_point_cloud_on_axis(
        ax0,
        point_cloud_0.points_3d,
        "Point Cloud from Stereo Pair 0",
        color="tab:blue",
    )

    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    plot_point_cloud_on_axis(
        ax1,
        point_cloud_1.points_3d,
        "Point Cloud from Stereo Pair 1",
        color="tab:orange",
    )

    plt.tight_layout()
    plt.show()

    return point_cloud_0, point_cloud_1


# ==========================================
# SECTION 3.2 HELPERS
# ==========================================
def plot_temporal_left_matches(
        left0_img: np.ndarray,
        kp_left0: list[cv2.KeyPoint],
        left1_img: np.ndarray,
        kp_left1: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
        num_to_draw: int = NUM_TEMPORAL_MATCHES_TO_DRAW,
) -> None:
    """Plots feature matches between left_0 and left_1."""
    matches_to_draw = matches[: min(num_to_draw, len(matches))]

    if left0_img.ndim == 2:
        left0_vis = cv2.cvtColor(left0_img, cv2.COLOR_GRAY2RGB)
    else:
        left0_vis = cv2.cvtColor(left0_img, cv2.COLOR_BGR2RGB)

    if left1_img.ndim == 2:
        left1_vis = cv2.cvtColor(left1_img, cv2.COLOR_GRAY2RGB)
    else:
        left1_vis = cv2.cvtColor(left1_img, cv2.COLOR_BGR2RGB)

    h0, w0 = left0_vis.shape[:2]
    h1, w1 = left1_vis.shape[:2]

    canvas_width = max(w0, w1)
    canvas_height = h0 + h1

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=left0_vis.dtype)
    canvas[:h0, :w0] = left0_vis
    canvas[h0:h0 + h1, :w1] = left1_vis

    plt.figure(figsize=(10, 12))
    plt.imshow(canvas)
    plt.axis("off")
    plt.title("Temporal feature matches between left_0 and left_1")

    for match in matches_to_draw:
        x0, y0 = kp_left0[match.queryIdx].pt
        x1, y1 = kp_left1[match.trainIdx].pt

        y1_shifted = y1 + h0

        plt.scatter([x0, x1], [y0, y1_shifted], s=20)
        plt.plot([x0, x1], [y0, y1_shifted], linewidth=1)

    plt.tight_layout()
    plt.show()


# ==========================================
# SECTION 3.2
# ==========================================
def section_3_2(
        frame_idx0: int = FRAME_0_INDEX,
        frame_idx1: int = FRAME_1_INDEX,
) -> tuple[np.ndarray, np.ndarray, list[cv2.DMatch]]:
    """Matches features between the two left images: left_0 and left_1."""
    print("--- Section 3.2 ---")

    left0_img, _ = read_images(frame_idx0)
    left1_img, _ = read_images(frame_idx1)

    kp_left0, kp_left1, matches = extract_and_match_features(left0_img, left1_img)
    left0_pts, left1_pts = get_matched_points(kp_left0, kp_left1, matches)

    print(f"Number of ratio-test matches between left_{frame_idx0} and left_{frame_idx1}: {len(matches)}")

    plot_temporal_left_matches(
        left0_img,
        kp_left0,
        left1_img,
        kp_left1,
        matches,
    )

    return left0_pts, left1_pts, matches


# ==========================================
# SECTION 3.3 HELPERS
# ==========================================
def get_k_matrix() -> np.ndarray:
    """Dynamically extracts the intrinsic matrix K from calib.txt."""
    P1, _ = read_calib()
    return P1[:, :3]


def plot_camera_positions(R: np.ndarray, tvec: np.ndarray) -> None:
    """Plots the X-Z bird's-eye view of the 4 cameras."""
    P1, P2 = read_calib()
    focal_length = P1[0, 0]
    tx_pixels = P2[0, 3]
    baseline = -tx_pixels / focal_length

    C_L0 = np.array([0, 0, 0])
    C_R0 = np.array([baseline, 0, 0])

    C_L1 = -R.T @ tvec.flatten()
    C_R1 = C_L1 + R.T @ np.array([baseline, 0, 0])

    plt.figure(figsize=(8, 6))

    cameras_x = [C_L0[0], C_R0[0], C_L1[0], C_R1[0]]
    cameras_z = [C_L0[2], C_R0[2], C_L1[2], C_R1[2]]
    labels = ['$left_0$', '$right_0$', '$left_1$', '$right_1$']
    colors = ['blue', 'blue', 'orange', 'orange']

    plt.scatter(cameras_x, cameras_z, c=colors, s=100, marker='s')

    for i, label in enumerate(labels):
        plt.annotate(
            label,
            (cameras_x[i], cameras_z[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    plt.plot([C_L0[0], C_R0[0]], [C_L0[2], C_R0[2]], 'b--', label='Stereo Pair 0')
    plt.plot([C_L1[0], C_R1[0]], [C_L1[2], C_R1[2]], color='orange', linestyle='--', label='Stereo Pair 1')
    plt.plot([C_L0[0], C_L1[0]], [C_L0[2], C_L1[2]], 'gray', linestyle=':', label='Camera Motion')

    plt.title("Relative Position of Four Cameras (Top-Down View)")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# ==========================================
# SECTION 3.3
# ==========================================
def section_3_3(
        point_cloud_0: StereoPointCloud,
        left0_pts: np.ndarray,
        left1_pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finds 4 key-points matched on all four images and calculates [R|t]."""
    print("--- Section 3.3 ---")

    K = get_k_matrix()

    # 1. Choose key-points matched on all four images using simple loops
    common_3d = []
    common_2d_left1 = []
    common_2d_left0 = []

    for i, pt_temporal in enumerate(left0_pts):
        for j, pt_stereo in enumerate(point_cloud_0.left_inliers):
            # If the pixel coordinates in left_0 match exactly, it's the same physical point
            if pt_temporal[0] == pt_stereo[0] and pt_temporal[1] == pt_stereo[1]:
                common_3d.append(point_cloud_0.points_3d[j])
                common_2d_left1.append(left1_pts[i])
                common_2d_left0.append(pt_temporal)
                break

    common_3d = np.array(common_3d)
    common_2d_left1 = np.array(common_2d_left1)
    common_2d_left0 = np.array(common_2d_left0)

    print(f"Found {len(common_3d)} common points across all four images.")

    if len(common_3d) < 4:
        raise ValueError("Not enough common points to compute PnP (need at least 4).")

    # 2. Choose 4 key-points
    random_indices = random.sample(range(len(common_3d)), 4)
    sample_3d = common_3d[random_indices]
    sample_2d = common_2d_left1[random_indices]

    # 3. Apply PNP algorithm
    success, rvec, tvec = cv2.solvePnP(sample_3d, sample_2d, K, None, flags=cv2.SOLVEPNP_EPNP)

    if not success:
        raise RuntimeError("cv2.solvePnP failed to find a solution.")

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    print(f"Calculated Extrinsic Matrix T:\n{np.round(T, 4)}")

    # 4. Plot cameras
    plot_camera_positions(R, tvec)

    return common_3d, common_2d_left1, common_2d_left0, T


# ==========================================
# MAIN EXECUTION
# ==========================================
def main() -> None:
    # Section 3.1
    point_cloud_0, point_cloud_1 = section_3_1()

    # Section 3.2
    left0_pts, left1_pts, temporal_matches = section_3_2()

    # Section 3.3
    common_3d, common_2d_left1, common_2d_left0, T = section_3_3(
        point_cloud_0,
        left0_pts,
        left1_pts
    )


if __name__ == "__main__":
    main()