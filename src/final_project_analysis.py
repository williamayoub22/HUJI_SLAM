import json
from pathlib import Path

import numpy as np

from src.slam.config import CACHE_DIR
from src.slam.database.facade import SlamDatabase


DATABASE_PATH = CACHE_DIR / "slam_db"
# SUMMARY_PATH = EX8_OUTPUT_DIR / "project_analysis_summary.json"


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


def run_tracking_analysis(database):
    """Compute the statistics required by the tracking-performance table."""
    manager_2d = database.manager_2d

    frame_ids = sorted(manager_2d.all_frames())
    track_ids = list(manager_2d.all_tracks())

    track_lengths = np.asarray(
        [len(manager_2d.frames(track_id)) for track_id in track_ids],
        dtype=float,
    )

    # Connectivity of frame i is the number of tracks that continue
    # from frame i to the following database frame.
    frame_connectivities = []

    for current_frame, next_frame in zip(frame_ids[:-1], frame_ids[1:]):
        current_tracks = set(manager_2d.tracks(current_frame))
        next_tracks = set(manager_2d.tracks(next_frame))

        frame_connectivities.append(
            len(current_tracks.intersection(next_tracks))
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
    }

    print("\n" + "=" * 60)
    print("Tracking database statistics")
    print("=" * 60)
    print(f"Number of frames:         {results['number_of_frames']}")
    print(f"Number of tracks:         {results['number_of_tracks']}")
    print(f"Mean track length:        {results['mean_track_length']:.2f}")
    print(
        "Mean frame connectivity: "
        f"{results['mean_frame_connectivity']:.2f}"
    )

    print("\nLaTeX table values:")
    print(
        f"Number of frames & {results['number_of_frames']} \\\\"
    )
    print(
        f"Number of tracks & {results['number_of_tracks']} \\\\"
    )
    print(
        f"Mean track length & "
        f"{results['mean_track_length']:.2f} \\\\"
    )
    print(
        f"Mean frame connectivity & "
        f"{results['mean_frame_connectivity']:.2f} \\\\"
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
    database = build_or_load_database()

    tracking_results = run_tracking_analysis(database)

    print(tracking_results)

    pnp_results = run_pnp_analysis(database)

    bundle_results = run_bundle_adjustment_analysis(database)

    pose_graph_results = run_pose_graph_analysis(database)

    loop_results = run_loop_closure_analysis(database)

    # save_summary_statistics({
    #     "tracking": tracking_results,
    #     "pnp": pnp_results,
    #     "bundle": bundle_results,
    #     "pose_graph": pose_graph_results,
    #     "loop_closure": loop_results,
    # })


if __name__ == "__main__":
    main()