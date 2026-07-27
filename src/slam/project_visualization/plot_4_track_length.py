"""
Provides plot 4 track length components and utilities for the SLAM pipeline.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
from src.slam.pipeline.caching import load_or_build_db


def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db = load_or_build_db()
    manager_2d = db.manager_2d
    track_ids = list(manager_2d.all_tracks())

    track_lengths = []
    for tid in tqdm(track_ids, desc="Track lengths"):
        track_lengths.append(len(manager_2d.frames(tid)))

    plt.figure(figsize=(10, 4))
    plt.hist(track_lengths, bins=range(2, 51), edgecolor="black", linewidth=0.5)
    plt.yscale("log")
    plt.xlim(0, 50)
    plt.xlabel("Track length histogram")
    plt.ylabel("# Tracks [logarithmic scale]")
    plt.title("Track Length Distribution")
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "4_track_length_histogram.png", dpi=150)


if __name__ == "__main__":
    main()
