"""
Provides plot 5 trajectory components and utilities for the SLAM pipeline.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.geometry.stereo import pose3_from_world_to_camera_extrinsic
from src.slam.pipeline.caching import load_or_build_db, load_or_build_pose_graph_with_lc


def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pg_data = load_or_build_pose_graph_with_lc()
    db = load_or_build_db()

    versions = pg_data["versions_data"]

    plt.figure(figsize=(10, 8))

    gt = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    gt_pos = []
    for g in gt:
        gt_pos.append(pose3_from_world_to_camera_extrinsic(g).translation())
    gt_pos = np.array(gt_pos)
    plt.plot(gt_pos[:, 0], gt_pos[:, 2], label="Ground Truth", color="black", linewidth=0.5)

    pnp_pos = []
    for fid in sorted(db.manager_poses.get_all_poses().keys()):
        p = pose3_from_world_to_camera_extrinsic(db.manager_poses.get_pose(fid)).translation()
        pnp_pos.append(p)
    pnp_pos = np.array(pnp_pos)
    plt.plot(pnp_pos[:, 0], pnp_pos[:, 2], label="PnP", alpha=0.7)

    no_lc = versions[0]["positions"]
    plt.plot(no_lc[:, 0], no_lc[:, 2], label="Pose Graph (No LC)", linestyle="--")

    with_lc = versions[-1]["positions"]
    plt.plot(with_lc[:, 0], with_lc[:, 2], label="Pose Graph (With LC)", linestyle="-.")

    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Trajectory Comparison")
    plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "5_trajectory.png", dpi=150)


if __name__ == "__main__":
    main()
