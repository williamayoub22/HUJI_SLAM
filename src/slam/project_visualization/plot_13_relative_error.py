import numpy as np
import matplotlib.pyplot as plt
from src.slam.pipeline.caching import load_or_build_pose_graph_with_lc, load_or_build_db
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.geometry.transforms import to_homogeneous_transform
from src.slam.geometry.stereo import pose3_from_world_to_camera_extrinsic
def relative_dists(poses, is_gtsam_matrix=False):
    dists = []
    for i in range(len(poses)-1):
        if is_gtsam_matrix:
            T1 = poses[i]
            T2 = poses[i+1]
            T_rel = np.linalg.inv(T1) @ T2
        else:
            T1 = to_homogeneous_transform(poses[i])
            T2 = to_homogeneous_transform(poses[i+1])
            T_rel = np.linalg.inv(T1) @ T2
        dists.append(np.linalg.norm(T_rel[:3,3]))
    return np.array(dists)

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    db = load_or_build_db()
    
    pnp = []
    for i in range(len(db.manager_poses.get_all_poses())): pnp.append(db.manager_poses.get_pose(i))
    
    pg_data = load_or_build_pose_graph_with_lc()
    pg = pg_data["pose_graph_matrices"]
    pg_lc = pg_data["pose_graph_lc_matrices"]
    
    from src.slam.pipeline.caching import load_or_build_pose_graph_no_lc
    keyframe_ids = load_or_build_pose_graph_no_lc()["keyframe_ids"]
    
    # We need to extract the world_to_camera extrinsics from GT
    gt_cam = [pose3_from_world_to_camera_extrinsic(g).matrix() for g in gt]
    
    gt_d = relative_dists(gt_cam, True)
    pnp_d = relative_dists(pnp, False)
    pg_d = relative_dists(pg, True)
    pg_lc_d = relative_dists(pg_lc, True)
    
    err_pnp = np.abs(pnp_d - gt_d[:len(pnp_d)])
    err_pg = np.abs(pg_d - gt_d[keyframe_ids[:-1]])
    err_pg_lc = np.abs(pg_lc_d - gt_d[keyframe_ids[:-1]])
    
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(err_pnp)), err_pnp, label="PnP", color="dodgerblue", alpha=0.5, linewidth=0.5)
    plt.plot(keyframe_ids[:-1], err_pg, label="PG (No LC)", color="darkorange", alpha=0.9, linewidth=0.5, linestyle="--")
    plt.plot(keyframe_ids[:-1], err_pg_lc, label="PG (With LC)", color="green", alpha=0.9, linewidth=0.5)
    plt.xlabel("Frame Pair")
    plt.ylabel("Relative Error [m]")
    plt.title("Relative Translation Error")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "13_relative_error.png", dpi=150)
if __name__ == "__main__": main()
