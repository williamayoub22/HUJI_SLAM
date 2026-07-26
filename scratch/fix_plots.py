import os

plot_10 = """import numpy as np
import matplotlib.pyplot as plt
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic
from src.ex8 import load_or_build_db
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
"""

plot_11 = """import numpy as np
import matplotlib.pyplot as plt
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic
from src.ex8 import load_or_build_pose_graph_no_lc
import gtsam

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    gt_poses = [pose3_from_world_to_camera_extrinsic(g) for g in gt]
    
    data = load_or_build_pose_graph_no_lc()
    est_matrices = data["pose_graph_matrices"]
    keyframe_ids = data["keyframe_ids"]
    
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
        
        R_err = est_poses[i].rotation().between(gt_poses[fid].rotation())
        err_angle.append(np.abs(R_err.axisAngle()[1]) * 180.0 / np.pi)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(keyframe_ids[:len(err_x)], err_x, label="x", linewidth=1)
    ax1.plot(keyframe_ids[:len(err_y)], err_y, label="y", linewidth=1)
    ax1.plot(keyframe_ids[:len(err_z)], err_z, label="z", linewidth=1)
    ax1.plot(keyframe_ids[:len(err_norm)], err_norm, label="norm", linewidth=1.5)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("Estimation Error (m)")
    ax1.set_title("Absolute Pose Graph (No LC) estimation error: Location")
    ax1.legend()
    ax1.grid()
    
    ax2.plot(keyframe_ids[:len(err_angle)], err_angle, color="red", linewidth=1.5)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("Angle Error (deg)")
    ax2.set_title("Absolute Pose Graph (No LC) estimation error: Angle")
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "11_absolute_pose_graph_no_lc_error.png", dpi=150)

if __name__ == "__main__": main()
"""

plot_12 = """import numpy as np
import matplotlib.pyplot as plt
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR, GT_POSES_PATH
from src.slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic
from src.ex8 import load_or_build_pose_graph_with_lc
import gtsam

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = np.loadtxt(GT_POSES_PATH).reshape(-1, 3, 4)
    gt_poses = [pose3_from_world_to_camera_extrinsic(g) for g in gt]
    
    data = load_or_build_pose_graph_with_lc()
    est_matrices = data["pose_graph_lc_matrices"]
    
    from src.ex8 import load_or_build_pose_graph_no_lc
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
        
        R_err = est_poses[i].rotation().between(gt_poses[fid].rotation())
        err_angle.append(np.abs(R_err.axisAngle()[1]) * 180.0 / np.pi)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(keyframe_ids[:len(err_x)], err_x, label="x", linewidth=1)
    ax1.plot(keyframe_ids[:len(err_y)], err_y, label="y", linewidth=1)
    ax1.plot(keyframe_ids[:len(err_z)], err_z, label="z", linewidth=1)
    ax1.plot(keyframe_ids[:len(err_norm)], err_norm, label="norm", linewidth=1.5)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("Estimation Error (m)")
    ax1.set_title("Absolute Pose Graph (With LC) estimation error: Location")
    ax1.legend()
    ax1.grid()
    
    ax2.plot(keyframe_ids[:len(err_angle)], err_angle, color="red", linewidth=1.5)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("Angle Error (deg)")
    ax2.set_title("Absolute Pose Graph (With LC) estimation error: Angle")
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "12_absolute_pose_graph_with_lc_error.png", dpi=150)

if __name__ == "__main__": main()
"""

with open("src/final_analysis/plot_10_absolute_pnp.py", "w") as f:
    f.write(plot_10)
with open("src/final_analysis/plot_11_absolute_pg_no_lc.py", "w") as f:
    f.write(plot_11)
with open("src/final_analysis/plot_12_absolute_pg_with_lc.py", "w") as f:
    f.write(plot_12)

print("Rewrote plots 10, 11, and 12")
