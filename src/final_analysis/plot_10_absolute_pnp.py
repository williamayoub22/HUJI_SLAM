import numpy as np
import matplotlib.pyplot as plt
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic
from src.slam.pipeline.caching import load_or_build_db
from src.slam.geometry.transforms import to_homogeneous_transform
import gtsam

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    
    gt_poses = [pose3_from_world_to_camera_extrinsic(g) for g in gt]
    
    db = load_or_build_db()
    est_poses = [pose3_from_world_to_camera_extrinsic(db.manager_poses.get_pose(fid)) for fid in range(len(db.manager_poses.get_all_poses()))]
    
    N = min(len(gt_poses), len(est_poses))
    gt_poses = gt_poses[:N]
    est_poses = est_poses[:N]
    
    err_x, err_y, err_z, err_norm = [], [], [], []
    err_angle = []
    
    for i in range(N):
        dt = est_poses[i].translation() - gt_poses[i].translation()
        err_x.append(np.abs(dt[0]))
        err_y.append(np.abs(dt[1]))
        err_z.append(np.abs(dt[2]))
        err_norm.append(np.linalg.norm(dt))
        
        # Angle error
        R_err = est_poses[i].rotation().between(gt_poses[i].rotation())
        err_angle.append(np.abs(R_err.axisAngle()[1]) * 180.0 / np.pi)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(err_x, label="x", linewidth=1)
    ax1.plot(err_y, label="y", linewidth=1)
    ax1.plot(err_z, label="z", linewidth=1)
    ax1.plot(err_norm, label="norm", linewidth=1.5)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("Estimation Error (m)")
    ax1.set_title("Absolute PnP estimation error: Location")
    ax1.legend()
    ax1.grid()
    
    ax2.plot(err_angle, color="red", linewidth=1.5)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("Angle Error (deg)")
    ax2.set_title("Absolute PnP estimation error: Angle")
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "10_absolute_pnp_error.png", dpi=150)

if __name__ == "__main__": main()
