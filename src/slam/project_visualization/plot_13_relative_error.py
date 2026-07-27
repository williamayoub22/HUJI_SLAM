import numpy as np
import matplotlib.pyplot as plt
import gtsam
from src.slam.pipeline.caching import load_or_build_pose_graph_with_lc, load_or_build_db
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.geometry.transforms import to_homogeneous_transform
from src.slam.geometry.stereo import pose3_from_world_to_camera_extrinsic

def make_pose3(T_4x4):
    """4x4 matrix (c2w or w2c) → gtsam.Pose3."""
    return gtsam.Pose3(T_4x4)

def relative_errors(poses, gt_poses_list, keyframe_ids=None):
    """Compute consecutive relative translation and rotation errors.

    If keyframe_ids is None, poses and gt_poses_list are assumed to be
    frame-aligned (PnP case). Otherwise poses[i] corresponds to gt_poses_list[keyframe_ids[i]].
    Returns (err_trans, err_angle_deg).
    """
    err_trans, err_angle = [], []
    n = len(poses) - 1
    for i in range(n):
        if keyframe_ids is not None:
            gt_i = gt_poses_list[keyframe_ids[i]]
            gt_j = gt_poses_list[keyframe_ids[i + 1]]
        else:
            gt_i = gt_poses_list[i]
            gt_j = gt_poses_list[i + 1]

        rel_est = make_pose3(poses[i]).between(make_pose3(poses[i + 1]))
        rel_gt  = gt_i.between(gt_j)
        err_pose = rel_est.between(rel_gt)

        err_trans.append(np.linalg.norm(err_pose.translation()))
        # Rodrigues axis-angle magnitude
        err_angle.append(np.linalg.norm(gtsam.Rot3.Logmap(err_pose.rotation())) * 180.0 / np.pi)
    return np.array(err_trans), np.array(err_angle)

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    db = load_or_build_db()

    # GT as gtsam.Pose3 (c2w)
    gt_poses = [gtsam.Pose3(pose3_from_world_to_camera_extrinsic(g).matrix()) for g in gt]

    # PnP poses as 4x4 c2w matrices
    pnp_raw = [db.manager_poses.get_pose(i) for i in range(len(db.manager_poses.get_all_poses()))]
    pnp_mats = [pose3_from_world_to_camera_extrinsic(p).matrix() for p in pnp_raw]

    pg_data = load_or_build_pose_graph_with_lc()
    pg    = pg_data["pose_graph_matrices"]    # list of 4x4, c2w, at keyframes
    pg_lc = pg_data["pose_graph_lc_matrices"]

    from src.slam.pipeline.caching import load_or_build_pose_graph_no_lc
    keyframe_ids = load_or_build_pose_graph_no_lc()["keyframe_ids"]

    pnp_mats_kf = [pnp_mats[i] for i in keyframe_ids]
    
    err_pnp_t,    err_pnp_a    = relative_errors(pnp_mats_kf, gt_poses, keyframe_ids=keyframe_ids)
    err_pg_t,     err_pg_a     = relative_errors(pg,          gt_poses, keyframe_ids=keyframe_ids)
    err_pg_lc_t,  err_pg_lc_a  = relative_errors(pg_lc,       gt_poses, keyframe_ids=keyframe_ids)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # --- Translation ---
    ax1.plot(keyframe_ids[:-1], err_pnp_t,
             label="PnP", color="dodgerblue", alpha=0.6, linewidth=1.0, zorder=1)
    ax1.plot(keyframe_ids[:-1], err_pg_t,
             label="PG (No LC)", color="darkorange", alpha=1.0, linewidth=2.0, linestyle="-", zorder=2)
    ax1.plot(keyframe_ids[:-1], err_pg_lc_t,
             label="PG (With LC)", color="green", alpha=1.0, linewidth=2.0, linestyle=":", zorder=3)
    ax1.set_xlabel("Frame Pair", fontsize=11)
    ax1.set_ylabel("Relative Translation Error [m]", fontsize=11)
    ax1.set_title("Relative Translation Error", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.4)

    # --- Angle (Rodrigues) ---
    ax2.plot(keyframe_ids[:-1], err_pnp_a,
             label="PnP", color="dodgerblue", alpha=0.6, linewidth=1.0, zorder=1)
    ax2.plot(keyframe_ids[:-1], err_pg_a,
             label="PG (No LC)", color="darkorange", alpha=1.0, linewidth=2.0, linestyle="-", zorder=2)
    ax2.plot(keyframe_ids[:-1], err_pg_lc_a,
             label="PG (With LC)", color="green", alpha=1.0, linewidth=2.0, linestyle=":", zorder=3)
    ax2.set_xlabel("Frame Pair", fontsize=11)
    ax2.set_ylabel("Relative Angle Error [deg]", fontsize=11)
    ax2.set_title("Relative Rotation Error", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "13_relative_error.png", dpi=200)

if __name__ == "__main__": main()
