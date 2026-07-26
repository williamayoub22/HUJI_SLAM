import matplotlib.pyplot as plt
from src.ex8 import load_or_build_loop_closures
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_build_loop_closures()["lc_stats"]
    if not data: return
    x = [d["target_frame"] for d in data]
    y = [d["inlier_percentage"] for d in data]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(x)), y)
    plt.xticks(range(len(x)), [f"{d['source_frame']}->{d['target_frame']}" for d in data], rotation=45)
    plt.ylabel("Inlier Percentage [%]")
    plt.title("Inlier Percentage per Loop Closure")
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "16_lc_inliers.png", dpi=150)
if __name__ == "__main__": main()
