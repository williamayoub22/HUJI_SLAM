import numpy as np
from dataclasses import dataclass
from src.ex8 import load_or_build_pose_graph_with_lc, load_or_build_pose_graph_no_lc
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.visualization.final_plots import plot_relative_error_subsections_line

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pg_data = load_or_build_pose_graph_with_lc()
    pose_graph_lc_matrices = pg_data["pose_graph_lc_matrices"]
    
    gt_extrinsics = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    frame_ids = load_or_build_pose_graph_no_lc()["keyframe_ids"]
    
    plot_relative_error_subsections_line(
        estimated_poses=pose_graph_lc_matrices,
        gt_poses=gt_extrinsics,
        frame_ids=frame_ids,
        title_prefix="Bundle",
        output_path=FINAL_ANALYSIS_OUTPUT_DIR / "14_relative_bundle_error_subsections.png",
        is_c2w_list=True,
    )

if __name__ == "__main__": main()
