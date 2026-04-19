import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, List

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
# --- Paths & File Structure ---
BASE_DIR = os.path.join('data', 'sequences', '00')
LEFT_IMG_DIR = 'image_0'
RIGHT_IMG_DIR = 'image_1'
IMG_FILENAME_FORMAT = '{:06d}.png'
FRAME_INDEX = 0

# --- Algorithm Parameters ---
NUM_FEATURES = 1000
RATIO_THRESHOLD = 0.75
KNN_NEIGHBORS = 2

# --- Visualization Parameters ---
RANDOM_SEED = 42
NUM_MATCHES_TO_DRAW = 20
NUM_DESCRIPTORS_TO_PRINT = 2
DOT_RADIUS = 15  # Increased size for better visibility
COLOR_KP = (0, 255, 0)  # Green (for keypoints)
COLOR_DOT = (0, 0, 255)  # Red in BGR (for single match highlight)
FIG_SIZE = (15, 5)  # Matplotlib figure dimensions


# ==========================================
# PART 0: Setup and Data Loading
# ==========================================
def read_images(idx: int = FRAME_INDEX):
    """Loads the stereo pair for a given index using pinned path formats."""
    img_name = IMG_FILENAME_FORMAT.format(idx)

    left_path = os.path.join(BASE_DIR, LEFT_IMG_DIR, img_name)
    right_path = os.path.join(BASE_DIR, RIGHT_IMG_DIR, img_name)

    img1 = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not read images. Check paths:\n{left_path}\n{right_path}")

    return img1, img2


# ==========================================
# PART 1.1: Detect and Extract Key-points
# ==========================================
def extract_features(img: np.ndarray, num_features: int = NUM_FEATURES):
    """LOGIC: Detects keypoints and extracts descriptors using SIFT."""
    sift = cv2.SIFT_create(nfeatures=num_features)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def vis_keypoints(img1: np.ndarray, img2: np.ndarray, kp1: list, kp2: list):
    """PDF: Plots the detected keypoints on both images."""
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=COLOR_KP, flags=0)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=COLOR_KP, flags=0)

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.suptitle('Key-points pixel locations')

    axes[0].imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Left Image')
    axes[1].imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Right Image')
    plt.show()


# ==========================================
# PART 1.2: Calculate Feature-descriptors
# ==========================================
def print_first_descriptors(des1: np.ndarray, num_to_print: int = NUM_DESCRIPTORS_TO_PRINT):
    """PDF: Prints the exact array values of the first few descriptors."""
    print(f"\n1.2: Descriptors of the first {num_to_print} features (Left Image):")
    for i in range(min(num_to_print, len(des1))):
        print(f"Descriptor {i + 1}:\n{des1[i]}")


# ==========================================
# PART 1.3: Match Descriptors
# ==========================================
def match_features_knn(des1: np.ndarray, des2: np.ndarray, k: int = KNN_NEIGHBORS):
    """LOGIC: Finds the k-nearest neighbors for each descriptor."""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=k)
    return knn_matches


def vis_matches(img1, kp1, img2, kp2, matches: list, num_to_draw: int = NUM_MATCHES_TO_DRAW, title: str = ""):
    """PDF: Plots a specified number of matches connecting the two images."""
    sample_size = min(num_to_draw, len(matches))
    random_matches = random.sample(matches, sample_size)

    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, random_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=FIG_SIZE)
    plt.title(title)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.show()


# ==========================================
# PART 1.4: Significance Test
# ==========================================
def filter_matches_ratio(knn_matches: list, ratio: float = RATIO_THRESHOLD):
    """LOGIC: Filters matches using Lowe's Ratio Test."""
    good_matches = []
    rejected_matches = []

    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
        else:
            rejected_matches.append(m)

    return good_matches, rejected_matches


def find_near_miss_match(knn_matches: list, ratio_threshold: float):
    """LOGIC: Finds the rejected match with the ratio closest to the threshold (most likely to be correct).

    this is for the very last question:
    "Present a correct match (as a dot on each image) that failed the significance test."
    """

    best_failed_match = None
    lowest_failed_ratio = float('inf')  # Start infinitely high

    for m, n in knn_matches:
        # Check if it failed the significance test
        if m.distance >= ratio_threshold * n.distance:
            current_ratio = m.distance / (n.distance + 1e-6)

            # Find the ratio that is just barely over the threshold
            if current_ratio < lowest_failed_ratio:
                lowest_failed_ratio = current_ratio
                best_failed_match = m

    return best_failed_match, lowest_failed_ratio


def vis_single_match_as_dots(img1, img2, kp1, kp2, match, title: str):
    """PDF: Plots a single match as large dots on both images for debugging/comparison."""
    pt1 = kp1[match.queryIdx].pt
    pt2 = kp2[match.trainIdx].pt

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.suptitle(title)

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    cv2.circle(img1_color, (int(pt1[0]), int(pt1[1])), DOT_RADIUS, COLOR_DOT, -1)
    cv2.circle(img2_color, (int(pt2[0]), int(pt2[1])), DOT_RADIUS, COLOR_DOT, -1)

    axes[0].imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Left Image Point: ({pt1[0]:.1f}, {pt1[1]:.1f})')
    axes[1].imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Right Image Point: ({pt2[0]:.1f}, {pt2[1]:.1f})')
    plt.show()


# ==========================================
# MAIN PIPELINE FUNCTION
# ==========================================
def find_features(frame_idx: int = FRAME_INDEX, visualize: bool = True) -> Tuple[
    List[cv2.KeyPoint], np.ndarray, List[cv2.KeyPoint], np.ndarray, List[cv2.DMatch]]:
    """Executes the full feature detection, description, and matching pipeline."""
    print(f"\n--- Processing Frame {frame_idx:06d} ---")
    random.seed(RANDOM_SEED)

    # --- Load Data ---
    img1, img2 = read_images(frame_idx)

    # --- Execute 1.1 ---
    kp1, des1 = extract_features(img1)
    kp2, des2 = extract_features(img2)
    print(f"1.1: Found {len(kp1)} keypoints in left, {len(kp2)} in right.")

    if visualize:
        vis_keypoints(img1, img2, kp1, kp2)

    # --- Execute 1.2 ---
    if visualize:
        print_first_descriptors(des1)

    # --- Execute 1.3 ---
    knn_matches = match_features_knn(des1, des2)

    if visualize:
        raw_matches_13 = [m[0] for m in knn_matches]
        vis_matches(img1, kp1, img2, kp2, raw_matches_13, title='1.3: 20 Random Matches (No significance test)')

    # --- Execute 1.4 ---
    good_matches, rejected_matches = filter_matches_ratio(knn_matches)

    if visualize:
        vis_matches(img1, kp1, img2, kp2, good_matches,
                    title=f"1.4: 20 Random Matches after Ratio Test (Ratio: {RATIO_THRESHOLD})")

        # Print 1.4 stats
        print(f"\n1.4 Statistics:")
        print(f"Ratio value used: {RATIO_THRESHOLD}")
        print(f"Total initial matches: {len(knn_matches)}")
        print(f"Matches discarded: {len(rejected_matches)}")
        print(f"Matches retained: {len(good_matches)}")

        # Visualize 1.4 failed match using the "Near Miss" logic
        failed_match, near_miss_ratio = find_near_miss_match(knn_matches, RATIO_THRESHOLD)
        if failed_match:
            print(f"\nFound the rejected match with the best ratio (Near miss): {near_miss_ratio:.4f}")
            vis_single_match_as_dots(img1, img2, kp1, kp2, failed_match,
                                     f'1.4: "Near Miss" Rejected Match (Ratio: {near_miss_ratio:.4f})')
        else:
            print("\nCould not find a rejected match.")

    return kp1, des1, kp2, des2, good_matches


# ==========================================
# EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    final_kp1, final_des1, final_kp2, final_des2, final_matches = find_features(FRAME_INDEX, visualize=True)
    print("\nPipeline execution complete.")
