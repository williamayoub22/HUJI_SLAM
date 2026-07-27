import numpy as np
import matplotlib.pyplot as plt
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.geometry.stereo import pose3_from_world_to_camera_extrinsic
from src.slam.pipeline.caching import load_or_build_pose_graph_with_lc
import gtsam

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    gt_poses = [pose3_from_world_to_camera_extrinsic(g) for g in gt]
    
    data = load_or_build_pose_graph_with_lc()
    est_matrices = data["pose_graph_lc_matrices"]
    
    from src.slam.pipeline.caching import load_or_build_pose_graph_no_lc
    keyframe_ids = load_or_build_pose_graph_no_lc()["keyframe_ids"]
    
    est_poses = [gtsam.Pose3(gtsam.Rot3(m[:3,:3]), gtsam.Point3(m[:3,3])) for m in est_matrices]
    
    err_x, err_y, err_z, err_norm = [], [], [], []
    err_angle = []
    
    for i, fid in enumerate(keyframe_ids):
        if fid >= len(gt_poses): break
        dt = est_poses[i].translation() - gt_poses[fid].translation()
        err_x.append(np.abs(dt[0]))
        err_y.append(np.abs(dt[1]))
        err_z.append(np.abs(dt[2]))
        err_norm.append(np.linalg.norm(dt))
        
        # Angle error: Rodrigues axis-angle magnitude (norm of logmap = theta)
        R_err = est_poses[i].rotation().between(gt_poses[fid].rotation())
        err_angle.append(np.linalg.norm(gtsam.Rot3.Logmap(R_err)) * 180.0 / np.pi)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(keyframe_ids[:len(err_x)], err_x, label="x", linewidth=0.5)
    ax1.plot(keyframe_ids[:len(err_y)], err_y, label="y", linewidth=0.5)
    ax1.plot(keyframe_ids[:len(err_z)], err_z, label="z", linewidth=0.5)
    ax1.plot(keyframe_ids[:len(err_norm)], err_norm, label="norm", linewidth=0.5)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("Estimation Error (m)")
    ax1.set_title("Absolute Pose Graph (With LC) estimation error: Location")
    ax1.legend()
    ax1.grid()
    
    ax2.plot(keyframe_ids[:len(err_angle)], err_angle, color="red", linewidth=0.5)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("Angle Error (deg)")
    ax2.set_title("Absolute Pose Graph (With LC) estimation error: Angle")
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "12_absolute_pose_graph_with_lc_error.png", dpi=150)

if __name__ == "__main__": main()
