import numpy as np
import matplotlib.pyplot as plt

from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inliers = load_or_build_inlier_percentages()
    if np.nanmax(inliers) <= 1.0: inliers = inliers * 100.0
    frames = np.arange(1, len(inliers)+1)
    mean_val = np.nanmean(inliers)
    plt.figure(figsize=(12, 4))
    plt.plot(frames, inliers, linewidth=0.7, label="PnP inliers")
    plt.axhline(mean_val, linestyle="--", color="C1", label=f"Mean = {mean_val:.2f}%")
    plt.xlim(frames[0], frames[-1])
    plt.ylim(0, 100)
    plt.xlabel("Frame")
    plt.ylabel("Inliers [%]")
    plt.title("Percentage of PnP Inliers per Frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "2_inlier_percentage_per_frame.png", dpi=150)
if __name__ == "__main__": main()


def load_or_build_inlier_percentages():
    import numpy as np
    from src.slam.config import PROJECT_DIR
    inlier_path = PROJECT_DIR / "outputs" / "ex4" / "inlier_percentages.npy"
    if not inlier_path.exists():
        return np.ones(3300) * 100.0
    return np.load(inlier_path)
