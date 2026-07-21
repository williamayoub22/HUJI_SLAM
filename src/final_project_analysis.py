import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.slam.config import CACHE_DIR, EX8_OUTPUT_DIR
from src.slam.database.facade import SlamDatabase


DATABASE_PATH = CACHE_DIR / "slam_db"
OUTPUT_DIR = EX8_OUTPUT_DIR
from src.slam.config import PROJECT_DIR

INLIER_PERCENTAGES_PATH = (
    PROJECT_DIR / "outputs" / "ex4" / "inlier_percentages.npy"
)

def build_or_load_database():
    """Load the tracking database created by the Exercise 8 pipeline."""
    database_file = Path(str(DATABASE_PATH) + ".pkl")

    # todo: change to load or rebuild
    if not database_file.exists():
        raise FileNotFoundError(
            f"Tracking database was not found at {database_file}. "
            "Run the database-building pipeline first."
        )

    print(f"Loading tracking database from {database_file}...")
    return SlamDatabase.load(str(DATABASE_PATH))


def plot_frame_connectivity(
    frame_ids,
    frame_connectivities,
    output_dir=OUTPUT_DIR,
):
    """Plot the number of tracks shared by each pair of consecutive frames."""
    print("\n" + "=" * 60)
    print("Connectivity")
    print("=" * 60)

    if not frame_connectivities:
        print(
            "WARNING: At least two frames are required to plot connectivity."
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Each connectivity value belongs to the transition
    # frame_ids[i] -> frame_ids[i + 1].
    source_frames = frame_ids[:-1]
    mean_connectivity = float(np.mean(frame_connectivities))

    plt.figure(figsize=(12, 4))
    plt.plot(
        source_frames,
        frame_connectivities,
        linewidth=0.7,
        label="Frame",
    )
    plt.axhline(
        y=mean_connectivity,
        linestyle="--",
        linewidth=1.0,
        label=f"Mean = {mean_connectivity:.2f}",
    )

    plt.xlabel("Frame")
    plt.ylabel("Outgoing tracks")
    plt.title("Connectivity")
    plt.xlim(source_frames[0], source_frames[-1])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "frame_connectivity.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Connectivity plot saved to {output_path}")
    return output_path


def plot_track_length_histogram(
    track_lengths,
    output_dir=OUTPUT_DIR,
):
    """Plot the distribution of track lengths on a logarithmic y-axis."""
    print("\n" + "=" * 60)
    print("Track length histogram")
    print("=" * 60)

    if track_lengths.size == 0:
        print("WARNING: No tracks are available for the histogram.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    integer_track_lengths = track_lengths.astype(int)

    # We display only lengths up to 50 because very long tracks are rare
    # and are less informative for visualizing the main distribution.
    max_display_length = 50
    bins = np.arange(0, max_display_length + 2)

    plt.figure(figsize=(12, 4))
    plt.hist(
        integer_track_lengths,
        bins=bins,
        log=True,
        edgecolor="black",
        linewidth=0.3,
    )

    plt.xlim(0, max_display_length)
    plt.xlabel("Track length histogram")
    plt.ylabel("# Tracks [logarithmic scale]")
    plt.title("Track Length Distribution")
    # plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    output_path = output_dir / "track_length_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Track-length histogram saved to {output_path}")
    return output_path


def plot_matches_per_frame(
    frame_ids,
    matches_per_frame,
    output_dir=OUTPUT_DIR,
):
    """Plot the number of feature matches stored in each frame."""
    print("\n" + "=" * 60)
    print("Matches per frame")
    print("=" * 60)

    if not frame_ids:
        print("WARNING: No frames are available.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    mean_matches = float(np.mean(matches_per_frame))

    plt.figure(figsize=(12, 4))
    plt.plot(
        frame_ids,
        matches_per_frame,
        linewidth=0.7,
        label="Matches",
    )
    plt.axhline(
        mean_matches,
        linestyle="--",
        linewidth=1.0,
        label=f"Mean = {mean_matches:.2f}",
    )

    plt.xlim(frame_ids[0], frame_ids[-1])
    plt.xlabel("Frame")
    plt.ylabel("Number of matches")
    plt.title("Number of Matches per Frame")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "matches_per_frame.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Matches-per-frame plot saved to {output_path}")
    return output_path


def plot_inlier_percentages(
    inlier_percentages,
    output_dir=OUTPUT_DIR,
):
    """Plot the PnP inlier percentage for every frame transition."""
    print("\n" + "=" * 60)
    print("Percentage of inliers per frame")
    print("=" * 60)

    if inlier_percentages is None or len(inlier_percentages) == 0:
        print(
            "WARNING: No inlier-percentage data are available. "
            "Rebuild the Exercise 4 database first."
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    percentages = np.asarray(inlier_percentages, dtype=float).reshape(-1)

    # Support values saved either as ratios [0, 1] or percentages [0, 100].
    if np.nanmax(percentages) <= 1.0:
        percentages = 100.0 * percentages

    # Value i describes the transition from frame i to frame i + 1.
    destination_frames = np.arange(1, len(percentages) + 1)
    mean_percentage = float(np.nanmean(percentages))

    plt.figure(figsize=(12, 4))
    plt.plot(
        destination_frames,
        percentages,
        linewidth=0.7,
        label="PnP inliers",
    )
    plt.axhline(
        y=mean_percentage,
        linestyle="--",
        linewidth=1.0,
        label=f"Mean = {mean_percentage:.2f}%",
    )

    plt.xlim(destination_frames[0], destination_frames[-1])
    plt.ylim(0, 100)
    plt.xlabel("Frame")
    plt.ylabel("Inliers [%]")
    plt.title("Percentage of PnP Inliers per Frame")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "inlier_percentage_per_frame.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Inlier-percentage plot saved to {output_path}")
    return output_path


def load_inlier_percentages():
    """Load the per-transition percentages generated in Exercise 4."""
    if not INLIER_PERCENTAGES_PATH.exists():
        print(
            f"WARNING: Inlier percentages were not found at "
            f"{INLIER_PERCENTAGES_PATH}.\n"
            "Run Exercise 4 with REBUILD_DB = True to generate the file."
        )
        return np.asarray([], dtype=float)

    percentages = np.asarray(
        np.load(INLIER_PERCENTAGES_PATH),
        dtype=float,
    ).reshape(-1)

    print(
        f"Loaded {len(percentages)} inlier percentages from "
        f"{INLIER_PERCENTAGES_PATH}"
    )

    return percentages


def run_tracking_analysis(database, inlier_percentages):
    """Compute and plot the tracking-performance statistics."""
    manager_2d = database.manager_2d

    frame_ids = sorted(manager_2d.all_frames())
    track_ids = list(manager_2d.all_tracks())

    track_lengths = np.asarray(
        [len(manager_2d.frames(track_id)) for track_id in track_ids],
        dtype=float,
    )

    # Number of stored feature observations/matches in every frame.
    matches_per_frame = [
        len(manager_2d.tracks(frame_id))
        for frame_id in frame_ids
    ]

    # Number of tracks shared by each pair of consecutive database frames.
    frame_connectivities = []

    for current_frame, next_frame in zip(frame_ids[:-1], frame_ids[1:]):
        current_tracks = set(manager_2d.tracks(current_frame))
        next_tracks = set(manager_2d.tracks(next_frame))

        frame_connectivities.append(
            len(current_tracks.intersection(next_tracks))
        )

    connectivity_plot_path = plot_frame_connectivity(
        frame_ids,
        frame_connectivities,
    )

    histogram_plot_path = plot_track_length_histogram(
        track_lengths,
    )
    #
    # matches_plot_path = plot_matches_per_frame(
    #     frame_ids,
    #     matches_per_frame,
    # )

    inliers_plot_path = plot_inlier_percentages(
        inlier_percentages,
    )

    results = {
        "number_of_frames": len(frame_ids),
        "number_of_tracks": len(track_ids),
        "mean_track_length": (
            float(np.mean(track_lengths))
            if track_lengths.size > 0
            else 0.0
        ),
        "mean_frame_connectivity": (
            float(np.mean(frame_connectivities))
            if frame_connectivities
            else 0.0
        ),
        "mean_matches_per_frame": (
            float(np.mean(matches_per_frame))
            if matches_per_frame
            else 0.0
        ),
        "mean_inlier_percentage": (
            float(
                np.mean(
                    100.0 * inlier_percentages
                    if (
                        len(inlier_percentages) > 0
                        and np.max(inlier_percentages) <= 1.0
                    )
                    else inlier_percentages
                )
            )
            if len(inlier_percentages) > 0
            else 0.0
        ),
        "frame_connectivities": frame_connectivities,
        "matches_per_frame": matches_per_frame,
        "connectivity_plot_path": (
            str(connectivity_plot_path)
            if connectivity_plot_path is not None
            else None
        ),
        "track_length_histogram_path": (
            str(histogram_plot_path)
            if histogram_plot_path is not None
            else None
        ),
        # "matches_per_frame_plot_path": (
        #     str(matches_plot_path)
        #     if matches_plot_path is not None
        #     else None
        # ),
        "inliers_per_frame_plot_path": (
            str(inliers_plot_path)
            if inliers_plot_path is not None
            else None
        ),
    }

    print("\n" + "=" * 60)
    print("Tracking database statistics")
    print("=" * 60)
    print(f"Number of frames:          {results['number_of_frames']}")
    print(f"Number of tracks:          {results['number_of_tracks']}")
    print(f"Mean track length:         {results['mean_track_length']:.2f}")
    print(
        f"Mean frame connectivity:  "
        f"{results['mean_frame_connectivity']:.2f}"
    )
    print(
        f"Mean matches per frame:    "
        f"{results['mean_matches_per_frame']:.2f}"
    )

    if len(inlier_percentages) > 0:
        print(
            f"Mean inlier percentage:    "
            f"{results['mean_inlier_percentage']:.2f}%"
        )

    print("\nLaTeX table values:")
    print(f"Number of frames & {results['number_of_frames']} \\\\")
    print(f"Number of tracks & {results['number_of_tracks']} \\\\")
    print(
        f"Mean track length & "
        f"{results['mean_track_length']:.2f} \\\\"
    )
    print(
        f"Mean frame connectivity & "
        f"{results['mean_frame_connectivity']:.2f} \\\\"
    )
    print(
        f"Mean matches per frame & "
        f"{results['mean_matches_per_frame']:.2f} \\\\"
    )

    if len(inlier_percentages) > 0:
        print(
            f"Mean inlier percentage & "
            f"{results['mean_inlier_percentage']:.2f}\\% \\\\"
        )

    return results

def run_pnp_analysis(database):
    pass


def run_bundle_adjustment_analysis(database):
    pass


def run_pose_graph_analysis(database):
    pass


def run_loop_closure_analysis(database):
    pass


# def save_summary_statistics(results):
#     """Save available analysis results as JSON."""
#     SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
#
#     with open(SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
#         json.dump(results, summary_file, indent=4)
#
#     print(f"\nSummary statistics saved to {SUMMARY_PATH}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    database = build_or_load_database()
    inlier_percentages = load_inlier_percentages()

    tracking_results = run_tracking_analysis(
        database,
        inlier_percentages,
    )

    print(tracking_results)

    pnp_results = run_pnp_analysis(database)
    bundle_results = run_bundle_adjustment_analysis(database)
    pose_graph_results = run_pose_graph_analysis(database)
    loop_results = run_loop_closure_analysis(database)

    print("\n" + "=" * 60)
    print(f"All analysis outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()