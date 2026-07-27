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

def main():
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
