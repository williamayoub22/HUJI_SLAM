import matplotlib.pyplot as plt
import numpy as np
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
from src.slam.pipeline.caching import load_or_build_inlier_percentages
from tqdm import tqdm

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inlier_percentages = load_or_build_inlier_percentages()
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(inlier_percentages)), inlier_percentages, linewidth=0.5)
    mean_val = np.mean(inlier_percentages) if len(inlier_percentages) > 0 else 0
    plt.axhline(mean_val, linestyle="--", color="C1", label=f"Mean: {mean_val:.1f}%")
    plt.xlabel("Frame ID")
    plt.ylabel("Inlier Percentage (%)")
    plt.title("Percentage of Inliers per Frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "2_inlier_percentage.png", dpi=150)

if __name__ == "__main__": main()
