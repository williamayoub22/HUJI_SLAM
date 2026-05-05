import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from utils.features import extract_features
from utils.matching import match_features
from utils.image_loader import read_images


FRAME_INDEX = 0
VERTICAL_DEVIATION_THRESHOLD = 2.0


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

    print(f"Total matches: {num_matches}")
    print(f"Matches with vertical deviation > {VERTICAL_DEVIATION_THRESHOLD} px: {num_bad}")
    print(f"Percentage: {percentage_bad:.2f}%")

    return {
        "left_img": left_img,
        "right_img": right_img,
        "kp_left": kp_left,
        "kp_right": kp_right,
        "matches": matches,
        "deviations": deviations,
    }


def main():
    section_2_1()


if __name__ == "__main__":
    main()