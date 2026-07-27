"""
Main entry point for generating all project visualizations and running the full SLAM pipeline.

This script sequentially executes all 18 plotting modules to generate the final analysis graphs.
It is completely unified and handles caching implicitly:
1. It first prints tracking database statistics matching Table 1 of the report.
2. It then iterates through all `plot_*` scripts.
3. Heavy computational results (like DB tracking, bundle adjustment, and pose graph optimization) 
   are automatically cached in the `outputs/cache/` directory so plotting is nearly instantaneous 
   on subsequent runs.

Outputs:
    Saves 18 high-resolution PNG plots to `outputs/final_analysis/`.
"""

from src.slam.project_visualization import (
    plot_1_matches,
    plot_2_inlier_percentage,
    plot_3_connectivity,
    plot_4_track_length,
    plot_5_trajectory,
    plot_6_optimization_mean_factor,
    plot_7_optimization_median_projection,
    plot_8_pnp_projection,
    plot_9_bundle_projection,
    plot_10_absolute_pnp,
    plot_11_absolute_pg_no_lc,
    plot_12_absolute_pg_with_lc,
    plot_13_relative_error,
    plot_14_relative_bundle_subsections,
    plot_15_lc_matches,
    plot_16_lc_inliers,
    plot_17_uncertainty,
)


def print_header(plot_num):
    print("\n" + "#" * 60)
    print(f"# Running plot {plot_num}...")
    print("#" * 60)


import numpy as np

from src.slam.pipeline.caching import load_or_build_db


def print_tracking_statistics():
    print("\n" + "=" * 50)
    print("Table 1: Tracking database statistics.")
    print("=" * 50)

    db = load_or_build_db()
    manager = db.manager_2d
    frames = sorted(manager.all_frames())
    num_frames = len(frames)

    tracks = manager.all_tracks()
    num_tracks = len(tracks)

    track_lengths = [len(manager.frames(t)) for t in tracks]
    mean_track_length = np.mean(track_lengths) if track_lengths else 0

    connectivity = []
    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i + 1]
        t1 = set(manager.tracks(f1))
        t2 = set(manager.tracks(f2))
        connectivity.append(len(t1.intersection(t2)))
    mean_connectivity = np.mean(connectivity) if connectivity else 0

    print(f"{'Statistic':<30} | {'Value':>15}")
    print("-" * 50)
    print(f"{'Number of frames':<30} | {num_frames:>15}")
    print(f"{'Number of tracks':<30} | {num_tracks:>15}")
    print(f"{'Mean track length':<30} | {mean_track_length:>15.2f}")
    print(f"{'Mean frame connectivity':<30} | {mean_connectivity:>15.2f}")
    print("=" * 50 + "\n")


def main():
    print("Computing general tracking statistics...")
    print_tracking_statistics()

    print_header(1)
    plot_1_matches.main()
    print_header(2)
    plot_2_inlier_percentage.main()
    print_header(3)
    plot_3_connectivity.main()
    print_header(4)
    plot_4_track_length.main()
    print_header(5)
    plot_5_trajectory.main()
    print_header(6)
    plot_6_optimization_mean_factor.main()
    print_header(7)
    plot_7_optimization_median_projection.main()
    print_header(8)
    plot_8_pnp_projection.main()
    print_header(9)
    plot_9_bundle_projection.main()
    print_header(10)
    plot_10_absolute_pnp.main()
    print_header(11)
    plot_11_absolute_pg_no_lc.main()
    print_header(12)
    plot_12_absolute_pg_with_lc.main()
    print_header(13)
    plot_13_relative_error.main()
    print_header(14)
    plot_14_relative_bundle_subsections.main()
    print_header(15)
    plot_15_lc_matches.main()
    print_header(16)
    plot_16_lc_inliers.main()
    print_header(17)
    plot_17_uncertainty.main()
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
