from slam.pipeline.database_pipeline import build_tracking_database
from slam.analysis.tracking_stats import compute_tracking_statistics
from slam.visualization.ex4_plots import (
    plot_connectivity_graph,
    plot_inlier_percentage,
    plot_track_length_histogram,
)
from slam.visualization.tracks import visualize_track_patches
from slam.analysis.reprojection_analysis import analyze_track_reprojection


def main():
    db, frame_inlier_percentages = build_tracking_database(...)

    stats = compute_tracking_statistics(db)
    print(stats)

    plot_connectivity_graph(db)
    plot_inlier_percentage(frame_inlier_percentages)
    plot_track_length_histogram(db)

    visualize_track_patches(db, track_length_at_least=6)

    analyze_track_reprojection(db, track_length_at_least=10)


if __name__ == "__main__":
    main()