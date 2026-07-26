from src.final_analysis import (
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

def main():
    print("Running plot 1...")
    plot_1_matches.main()
    print("Running plot 2...")
    plot_2_inlier_percentage.main()
    print("Running plot 3...")
    plot_3_connectivity.main()
    print("Running plot 4...")
    plot_4_track_length.main()
    print("Running plot 5...")
    plot_5_trajectory.main()
    print("Running plot 6...")
    plot_6_optimization_mean_factor.main()
    print("Running plot 7...")
    plot_7_optimization_median_projection.main()
    print("Running plot 8...")
    plot_8_pnp_projection.main()
    print("Running plot 9...")
    plot_9_bundle_projection.main()
    print("Running plot 10...")
    plot_10_absolute_pnp.main()
    print("Running plot 11...")
    plot_11_absolute_pg_no_lc.main()
    print("Running plot 12...")
    plot_12_absolute_pg_with_lc.main()
    print("Running plot 13...")
    plot_13_relative_error.main()
    print("Running plot 14...")
    plot_14_relative_bundle_subsections.main()
    print("Running plot 15...")
    plot_15_lc_matches.main()
    print("Running plot 16...")
    plot_16_lc_inliers.main()
    print("Running plot 17...")
    plot_17_uncertainty.main()
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
