import numpy as np
from src.slam.pipeline.caching import load_or_build_pose_graph_with_lc, load_or_build_pose_graph_no_lc, load_or_build_subsection_pairs
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.project_visualization.plot_helpers import plot_relative_error_subsections_line

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pg_data = load_or_build_pose_graph_with_lc()
    pose_graph_lc_matrices = pg_data["pose_graph_lc_matrices"]
    
    gt_extrinsics = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    subsection_pairs = load_or_build_subsection_pairs(lengths=(100, 400, 800))
    
    plot_relative_error_subsections_line(
        estimated_poses=pose_graph_lc_matrices,
        gt_poses=gt_extrinsics,
        subsection_pairs=subsection_pairs,
        title_prefix="Bundle",
        output_path=FINAL_ANALYSIS_OUTPUT_DIR / "14_relative_bundle_error_subsections.png",
        is_c2w_list=True,
    )

if __name__ == "__main__": main()
