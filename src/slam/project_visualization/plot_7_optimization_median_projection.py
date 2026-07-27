"""
Provides plot 7 optimization median projection components and utilities for the SLAM pipeline.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
from src.slam.pipeline.caching import load_or_build_bundle_windows


def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_build_bundle_windows()
    solutions = data["context"].solutions

    window_ids = []
    initial_median_proj_errors = []
    final_median_proj_errors = []

    for window_index, solution in tqdm(
        enumerate(solutions, start=1), total=len(solutions), desc="Median projection errors"
    ):
        result = solution.result
        if result.num_factors <= 0:
            continue

        init_proj_errors = []
        final_proj_errors = []
        for metadata in result.projection_factor_metadata:
            factor = result.graph.at(metadata.factor_index)
            e_init = factor.unwhitenedError(result.initial)
            e_final = factor.unwhitenedError(result.optimized)
            init_proj_errors.append(np.linalg.norm(e_init))
            final_proj_errors.append(np.linalg.norm(e_final))

        if not init_proj_errors:
            continue

        window_ids.append(window_index)
        initial_median_proj_errors.append(np.median(init_proj_errors))
        final_median_proj_errors.append(np.median(final_proj_errors))

    plt.figure(figsize=(12, 5))
    plt.plot(window_ids, initial_median_proj_errors, label="Initial Error", color="orange")
    plt.plot(window_ids, final_median_proj_errors, label="Optimized Error", color="blue")
    plt.xlabel("Bundle Window Index")
    plt.ylabel("Median Projection Error")
    plt.title("Median Projection Error Before and After Bundle Optimization")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "7_median_projection_error.png", dpi=150)


if __name__ == "__main__":
    main()
