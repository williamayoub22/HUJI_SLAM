"""
Provides plot 1 matches components and utilities for the SLAM pipeline.
"""

import matplotlib.pyplot as plt
from tqdm import tqdm

from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
from src.slam.pipeline.caching import load_or_build_db


def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db = load_or_build_db()
    manager_2d = db.manager_2d
    frame_ids = sorted(manager_2d.all_frames())

    matches_per_frame = []
    for fid in tqdm(frame_ids, desc="Matches per frame"):
        matches_per_frame.append(len(manager_2d.tracks(fid)))

    plt.figure(figsize=(10, 4))
    plt.plot(frame_ids, matches_per_frame, linewidth=0.5)
    mean_val = sum(matches_per_frame) / len(matches_per_frame) if matches_per_frame else 0
    plt.axhline(mean_val, linestyle="--", color="C1", label=f"Mean: {mean_val:.1f}")
    plt.xlabel("Frame ID")
    plt.ylabel("Number of Matches")
    plt.title("Matches per Frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "1_matches_per_frame.png", dpi=150)


if __name__ == "__main__":
    main()
