import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from slam.config import PROJECT_DIR
from slam.io.image_loader import read_images
from slam.tracking_database import TrackingDB

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = PROJECT_DIR / "outputs" / "ex4"
DB_PATH = OUTPUT_DIR / "tracking_db"


def main():
    print("=" * 60)
    print("  Debugging Zero Connectivity")
    print("=" * 60)

    pkl_path = Path(str(DB_PATH) + ".pkl")
    if not pkl_path.exists():
        print(f"Tracking database not found at {pkl_path}")
        return

    print(f"Loading existing database from {pkl_path} ...")
    db = TrackingDB()
    db.load(str(DB_PATH))

    all_frames = sorted(db.all_frames())
    zero_connectivity_frames = []

    print("Computing outgoing tracks...")
    for fid in tqdm(all_frames[:-1]):
        tracks_cur = set(db.tracks(fid))
        tracks_next = set(db.tracks(fid + 1))
        if len(tracks_cur & tracks_next) == 0:
            zero_connectivity_frames.append(fid)

    print(
        f"Found {len(zero_connectivity_frames)} frames with zero connectivity: {zero_connectivity_frames}"
    )

    from slam.features.matching import match_and_filter

    # Plot the frames with zero connectivity
    for fid in zero_connectivity_frames:
        print(f"Processing zero-connectivity frame {fid} -> {fid + 1}")
        left_img_cur, _ = read_images(fid)
        left_img_next, _ = read_images(fid + 1)

        # We need 4 copies of images for the 2x2 grid
        def to_bgr(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img.copy()

        img_cur_inliers = to_bgr(left_img_cur)
        img_next_inliers = to_bgr(left_img_next)
        img_cur_lowe = to_bgr(left_img_cur)
        img_next_lowe = to_bgr(left_img_next)

        # 1. Draw Inliers (from tracks)
        tracks_cur = db.tracks(fid)
        for t_id in tracks_cur:
            xl, xr, y = db.link_triplet(fid, t_id)
            cv2.circle(img_cur_inliers, (int(xl), int(y)), 3, (0, 0, 255), -1)

        tracks_next = db.tracks(fid + 1)
        for t_id in tracks_next:
            xl, xr, y = db.link_triplet(fid + 1, t_id)
            cv2.circle(img_next_inliers, (int(xl), int(y)), 3, (0, 0, 255), -1)

        # 2. Draw Lowe's ratio test matches
        des_cur = db.features(fid)
        des_next = db.features(fid + 1)
        links_cur = db.all_frame_links(fid)
        links_next = db.all_frame_links(fid + 1)

        good_matches = []
        if des_cur is not None and des_next is not None and len(des_cur) > 0 and len(des_next) > 0:
            good_matches = match_and_filter(des_cur, des_next, feature_type="akaze", ratio=0.85)

            for m in good_matches:
                link_cur = links_cur[m.queryIdx]
                link_next = links_next[m.trainIdx]
                cv2.circle(
                    img_cur_lowe, (int(link_cur.x_left), int(link_cur.y)), 3, (0, 255, 0), -1
                )
                cv2.circle(
                    img_next_lowe, (int(link_next.x_left), int(link_next.y)), 3, (0, 255, 0), -1
                )

        # Plot 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            f"Zero Connectivity: Frame {fid} to Frame {fid + 1}\n"
            f"Inliers: {len(tracks_cur)} (F{fid}), {len(tracks_next)} (F{fid + 1}) | "
            f"Lowe's Matches: {len(good_matches)}"
        )

        axes[0, 0].imshow(cv2.cvtColor(img_cur_inliers, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Frame {fid} (RANSAC Inliers)")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(img_next_inliers, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"Frame {fid + 1} (RANSAC Inliers)")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(cv2.cvtColor(img_cur_lowe, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"Frame {fid} (Lowe's Ratio Matches)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(cv2.cvtColor(img_next_lowe, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"Frame {fid + 1} (Lowe's Ratio Matches)")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
