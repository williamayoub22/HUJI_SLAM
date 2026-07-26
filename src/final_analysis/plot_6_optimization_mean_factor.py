import matplotlib.pyplot as plt
import numpy as np
from src.ex8 import load_or_build_bundle_windows
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
from tqdm import tqdm

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_build_bundle_windows()
    solutions = data["context"].solutions
    
    window_ids = []
    initial_average_errors = []
    final_average_errors = []
    
    for window_index, solution in tqdm(enumerate(solutions, start=1), total=len(solutions), desc="Mean factor errors"):
        result = solution.result
        if result.num_factors <= 0: continue
        initial_error = float(result.average_initial_error)
        final_error = float(result.average_final_error)
        if not np.isfinite(initial_error) or not np.isfinite(final_error): continue
        
        window_ids.append(window_index)
        initial_average_errors.append(initial_error)
        final_average_errors.append(final_error)
        
    plt.figure(figsize=(12, 5))
    plt.plot(window_ids, initial_average_errors, label="Initial Error", color="orange")
    plt.plot(window_ids, final_average_errors, label="Optimized Error", color="blue")
    plt.xlabel("Bundle Window Index")
    plt.ylabel("Mean Factor Error")
    plt.title("Mean Factor Error Before and After Bundle Optimization")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "6_mean_factor_error.png", dpi=150)

if __name__ == "__main__": main()
