"""
Provides plot 8 pnp projection components and utilities for the SLAM pipeline.
"""

import matplotlib.pyplot as plt
from tqdm import tqdm

from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
from src.slam.pipeline.caching import load_or_build_projection_data


def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Using the cached data since computing it takes a very long time.
    # The progress of the plot creation itself is fast.
    with tqdm(total=1, desc="Loading PnP projection data") as pbar:
        data = load_or_build_projection_data()["pnp_projection_data"]
        pbar.update(1)

    plt.figure(figsize=(10, 5))
    plt.plot(data["distances"], data["median_left_errors"], label="Left", marker="o", markersize=3)
    plt.plot(
        data["distances"], data["median_right_errors"], label="Right", marker="o", markersize=3
    )
    plt.xlabel("Distance from Triangulation [frames]")
    plt.ylabel("Median Projection Error [px]")
    plt.title("PnP Projection Error vs Distance")
    plt.xlim(0, 50)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "8_pnp_projection_error_vs_distance.png", dpi=150)


if __name__ == "__main__":
    main()
