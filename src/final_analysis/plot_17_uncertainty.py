import numpy as np
import matplotlib.pyplot as plt
from src.slam.pipeline.caching import load_or_build_pose_graph_with_lc
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_build_pose_graph_with_lc()
    versions = data["versions_data"]
    
    cov_no_lc = versions[0]["covariances"]
    cov_lc = versions[-1]["covariances"]
    
    kf_ids = sorted(cov_no_lc.keys())
    
    uncert_no_lc = []
    uncert_lc = []
    for fid in kf_ids:
        c1 = cov_no_lc[fid][3:6, 3:6]
        c2 = cov_lc[fid][3:6, 3:6]
        uncert_no_lc.append(np.linalg.det(c1))
        uncert_lc.append(np.linalg.det(c2))
        
    plt.figure(figsize=(10, 5))
    plt.plot(kf_ids, uncert_no_lc, label="Without Loop Closures", color="red")
    plt.plot(kf_ids, uncert_lc, label="With Loop Closures", color="blue")

    plt.title("Location Uncertainty Size (det(Cov_translation))")
    plt.xlabel("Frame ID")
    plt.ylabel("Determinant of Translation Covariance")
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "17_uncertainty.png", dpi=200)

if __name__ == "__main__": main()
