import numpy as np
from src.slam.pipeline.caching import load_or_build_pose_graph_with_lc, load_or_build_loop_closures
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
from src.slam.project_visualization.plot_helpers import plot_uncertainty

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_build_pose_graph_with_lc()
    versions = data["versions_data"]

    lc_data = load_or_build_loop_closures()
    lc_stats = lc_data.get("lc_stats", [])

    # versions[0] = Optimized Without LC, versions[-1] = Optimized with All LCs
    keyframe_ids = sorted(versions[0]["covariances"].keys()) if versions[0]["covariances"] else []

    plot_uncertainty(
        versions_data=[versions[0], versions[-1]],
        lc_stats=lc_stats,
        keyframe_ids=keyframe_ids,
        output_path=FINAL_ANALYSIS_OUTPUT_DIR / "17_uncertainty.png",
    )

if __name__ == "__main__": main()
