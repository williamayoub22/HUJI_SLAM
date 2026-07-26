from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.slam.analysis.tracking_stats import compute_tracking_statistics, get_track_lengths
from src.slam.config import PROJECT_DIR
from src.slam.geometry.projection import project_points
from src.slam.geometry.triangulation import triangulate_opencv
from src.slam.io.calibration import read_stereo_calibration
from src.slam.io.image_loader import read_images
from src.slam.io.poses import read_ground_truth_poses
from src.slam.pipeline.database_pipeline import build_tracking_database
from src.slam.database.tracking_database import TrackingDB
from tqdm import tqdm

SECTION_7_MIN_TRACKS_LENGTH = 10

# ============================================================================
# CONFIGURATION — change these to control behaviour
# ============================================================================

# Number of frames to process.  Set to None for the full sequence.
NUM_FRAMES = None
REBUILD_DB = True

SEQUENCE_DIR = PROJECT_DIR / "dataset" / "sequences" / "00"
POSES_PATH = PROJECT_DIR / "dataset" / "poses" / "00.txt"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ex4"
DB_PATH = OUTPUT_DIR / "tracking_db"


# ############################################################################
#                          SECTION 4.1 — Database
# ############################################################################


def build_or_load_database():
    """Section 4.1: Build the tracking database (or load from disk).

    Returns:
    -------
    db : TrackingDB
    inlier_percentages : list[float]
        Per-frame-transition percentage of PnP inliers (empty if loaded).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = Path(str(DB_PATH) + ".pkl")
    inlier_npy = OUTPUT_DIR / "inlier_percentages.npy"

    if REBUILD_DB or not pkl_path.exists():
        print("[4.1] Building tracking database from scratch …")
        db, inlier_percentages = build_tracking_database(
            sequence_dir=SEQUENCE_DIR,
            num_frames=NUM_FRAMES,
            feature_type="akaze",
            num_features=3000,
            ratio_threshold=0.85,
            deviation_threshold=2.0,
        )
        print("[4.1] Running consistency check …")
        db._check_consistency()

        print("[4.1] Serializing database …")
        db.serialize(str(DB_PATH))
        np.save(str(inlier_npy), np.array(inlier_percentages))
        print(f"[4.1] Database saved to {DB_PATH}")
    else:
        print(f"[4.1] Loading existing database from {pkl_path} …")
        db = TrackingDB()
        db.load(str(DB_PATH))
        if inlier_npy.exists():
            inlier_percentages = np.load(str(inlier_npy)).tolist()
        else:
            inlier_percentages = []
        print(f"[4.1] Database loaded — {db.frame_num()} frames, {db.track_num()} tracks")

    return db, inlier_percentages


# ############################################################################
#              SECTION 4.2 — Tracking Statistics
# ############################################################################
def compute_and_print_statistics(db: TrackingDB):
    """Section 4.2: Compute and print tracking statistics.

    Uses the existing compute_tracking_statistics() from tracking_stats.py.
    The TrackingDB guarantees all tracks have length >= 2, so no trivial
    (length-1) tracks exist.
    """
    print("\n" + "=" * 60)
    print("[4.2] Computing tracking statistics …")
    print("=" * 60)

    stats = compute_tracking_statistics(db)
    print(stats)

    # Also collect track lengths for use in section 4.6 histogram
    track_lengths = np.array(get_track_lengths(db))

    return track_lengths


# ############################################################################
#  SECTION 4.3 — Display a Track of Length >= 6
# ############################################################################
def display_track_length_6(db: TrackingDB):
    """Section 4.3: Pick a track of length >= 6 and show 20×20 crops
    around the feature on each left image, plus the full image with
    the feature marked.
    """
    print("\n" + "=" * 60)
    print("[4.3] Displaying a track of length >= 6 …")
    print("=" * 60)

    track_id = db.first_track_with_length_range(6, 8)
    if not track_id:
        print("[4.3] WARNING: No tracks of length >= 6 found.")
        return

    if track_id is None:
        print("[4.3] WARNING: No track with 6 <= length < 8 found.")
        return

    frame_ids = db.frames(track_id)
    print(f"[4.3] Selected track #{track_id} — appears in {len(frame_ids)} frames: {frame_ids}")

    n = len(frame_ids)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Track #{track_id}, length {len(frame_ids)}", fontsize=14)
    for i, fid in enumerate(tqdm(frame_ids, desc="[4.3] Loading frames")):
        left_img, _ = read_images(fid)
        xl, xr, y = db.link_triplet(fid, track_id)

        if len(left_img.shape) == 2:
            img_colour = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
        else:
            img_colour = left_img.copy()

        # 20×20 crop around feature (clamped to image boundaries)
        h, w = left_img.shape[:2]
        x_min = max(0, int(xl) - 10)
        x_max = min(w, int(xl) + 10)
        y_min = max(0, int(y) - 10)
        y_max = min(h, int(y) + 10)
        crop = img_colour[y_min:y_max, x_min:x_max].copy()

        # Mark the feature on the crop
        cx = int(xl) - x_min
        cy = int(y) - y_min
        cv2.drawMarker(
            crop, (cx, cy), (0, 0, 255), markerType=cv2.LINE_AA, markerSize=3, thickness=1
        )

        # Mark the feature on the full image
        img_marked = img_colour.copy()
        cv2.circle(img_marked, (int(xl), int(y)), 7, (0, 0, 255), -1)

        # --- plot full image ---
        axes[i, 0].imshow(cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"track #{track_id}, frame #{fid}", fontsize=9)
        axes[i, 0].axis("off")

        # --- plot crop ---
        axes[i, 1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title("20x20 crop", fontsize=9)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / "q4_3_track_display.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"[4.3] Saved to {out}")


# ############################################################################
#  SECTION 4.4 — Connectivity Graph
# ############################################################################


def plot_connectivity(db: TrackingDB):
    """Section 4.4: For each frame, plot the number of tracks that also
    appear in the *next* frame (outgoing tracks).
    """
    print("\n" + "=" * 60)
    print("[4.4] Computing connectivity graph …")
    print("=" * 60)

    all_frames = sorted(db.all_frames())
    outgoing_counts = []

    for fid in tqdm(all_frames[:-1], desc="[4.4] Computing outgoing tracks"):
        tracks_cur = set(db.tracks(fid))
        tracks_next = set(db.tracks(fid + 1))
        outgoing_counts.append(len(tracks_cur & tracks_next))

    frames_plot = all_frames[:-1]
    mean_val = np.mean(outgoing_counts)

    plt.figure(figsize=(12, 4))
    plt.plot(frames_plot, outgoing_counts, linewidth=0.7)
    plt.axhline(y=mean_val, color="g", linestyle="-", linewidth=1.0)
    plt.xlim(0, 3300)
    plt.xlabel("frame")
    plt.ylabel("outgoing tracks")
    plt.title("Connectivity")
    plt.tight_layout()

    out = OUTPUT_DIR / "q4_4_connectivity.png"
    plt.savefig(str(out), dpi=150)
    plt.close()


# ############################################################################
#  SECTION 4.5 — Percentage of Inliers per Frame
# ############################################################################


def plot_inlier_percentages(inlier_percentages):
    """Section 4.5: Plot percentage of PnP inliers per frame transition."""
    print("\n" + "=" * 60)
    print("[4.5] Plotting inlier percentages …")
    print("=" * 60)

    if not inlier_percentages:
        print(
            "[4.5] WARNING: No inlier data available (database was loaded, not rebuilt).  Skipping."
        )
        return

    frames = list(range(1, len(inlier_percentages) + 1))

    plt.figure(figsize=(12, 4))
    plt.plot(frames, inlier_percentages, linewidth=0.7)
    plt.xlabel("frame")
    plt.ylabel("% inliers")
    plt.title("Percentage of inliers per frame")
    plt.tight_layout()
    out = OUTPUT_DIR / "q4_5_inlier_pct.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"[4.5] Saved to {out}")


# ############################################################################
#  SECTION 4.6 — Track-Length Histogram
# ############################################################################


def plot_track_length_histogram(track_lengths):
    """Section 4.6: Histogram of track lengths (log scale y-axis).
    ``track_lengths`` should already exclude trivial (length-1) tracks.
    """
    print("\n" + "=" * 60)
    print("[4.6] Plotting track-length histogram …")
    print("=" * 60)

    if len(track_lengths) == 0:
        print("[4.6] WARNING: No tracks to plot.")
        return

    max_len = int(np.max(track_lengths))

    plt.figure(figsize=(12, 4))
    plt.hist(track_lengths, bins=range(2, max_len + 2), log=True, edgecolor="black", linewidth=0.3)
    plt.xlabel("Track length")
    plt.ylabel("Track #")
    plt.title("Track length histogram")
    plt.tight_layout()
    out = OUTPUT_DIR / "q4_6_track_hist.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"[4.6] Saved to {out}")


# ############################################################################
#  SECTION 4.7 — Reprojection Error
# ############################################################################


def plot_reprojection_error(db: TrackingDB):
    """Section 4.7: Pick a random track of length >= 10.  Triangulate its 3D
    point from the *last* frame using GT poses, project it back to all
    frames, and plot L2 reprojection error vs distance from reference.
    """
    print("\n" + "=" * 60)
    print("[4.7] Computing reprojection error …")
    print("=" * 60)

    tracks_10 = db.tracks_with_min_length(SECTION_7_MIN_TRACKS_LENGTH)
    if not tracks_10:
        print("[4.7] WARNING: No tracks of length >= 10 found.  Skipping.")
        return

    idx = np.random.randint(0, len(tracks_10))
    track_id = tracks_10[idx]
    frame_ids = db.frames(track_id)
    print(
        f"[4.7] Selected track #{track_id} — length {len(frame_ids)}, "
        f"frames {frame_ids[0]}–{frame_ids[-1]}"
    )

    # Read GT poses and calibration
    print("[4.7] Loading ground-truth poses …")
    poses = read_ground_truth_poses(POSES_PATH)
    P1, P2 = read_stereo_calibration()

    def pose_to_4x4(pose_3x4):
        """Expand a KITTI 3×4 pose to a 4×4 homogeneous transform."""
        T = np.eye(4)
        T[:3, :] = pose_3x4
        return T

    # Triangulate from the LAST frame
    last_fid = frame_ids[-1]
    T_last = pose_to_4x4(poses[last_fid])
    xl, xr, y = db.link_triplet(last_fid, track_id)
    point_3d = triangulate_opencv(
        P1 @ T_last,
        P2 @ T_last,
        np.array([[xl, y]]),
        np.array([[xr, y]]),
    )  # shape (1, 3)

    # Project to all frames and compute reprojection error
    errs_left = []
    errs_right = []
    distances = []

    for fid in tqdm(frame_ids, desc="[4.7] Projecting to each frame"):
        T_f = pose_to_4x4(poses[fid])

        proj_l = project_points(point_3d, P1 @ T_f)[0]
        proj_r = project_points(point_3d, P2 @ T_f)[0]

        xl_t, xr_t, y_t = db.link_triplet(fid, track_id)

        errs_left.append(np.linalg.norm(proj_l - np.array([xl_t, y_t])))
        errs_right.append(np.linalg.norm(proj_r - np.array([xr_t, y_t])))
        distances.append(last_fid - fid)

    distances = np.array(distances)
    errs_left = np.array(errs_left)
    errs_right = np.array(errs_right)

    order = np.argsort(distances)

    plt.figure(figsize=(10, 5))
    plt.plot(distances[order], errs_left[order], label="Left")
    plt.plot(distances[order], errs_right[order], label="Right")
    plt.xlabel("distance from reference")
    plt.ylabel("projection error (pixels)")
    plt.title("PnP - projection error vs track length")
    plt.legend()
    plt.tight_layout()
    out = OUTPUT_DIR / "q4_7_reprojection_error.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"[4.7] Saved to {out}")


def main():
    print("=" * 60)
    print("  Exercise 4 — Vision Aided Navigation 2026")
    print("=" * 60)
    if NUM_FRAMES is not None:
        print(f"  Processing {NUM_FRAMES} frames (set NUM_FRAMES = None for full sequence)")
    else:
        print("  Processing ALL frames")
    print("=" * 60 + "\n")

    # 4.1 — Build / load the tracking database
    db, inlier_percentages = build_or_load_database()

    # 4.2 — Tracking statistics (excludes trivial length-1 tracks)
    track_lengths = compute_and_print_statistics(db)

    # 4.3 — Display a track of length >= 6
    display_track_length_6(db)

    # 4.4 — Connectivity graph
    plot_connectivity(db)

    # 4.5 — Inlier percentages per frame
    plot_inlier_percentages(inlier_percentages)

    # 4.6 — Track-length histogram
    plot_track_length_histogram(track_lengths)

    # 4.7 — Reprojection error for a track of length >= 10
    plot_reprojection_error(db)

    print("\n" + "=" * 60)
    print(f"  All outputs saved to:  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
