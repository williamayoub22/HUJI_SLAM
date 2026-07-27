import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
def positions_from_world_to_camera_extrinsics(extrinsics_list, frame_ids):
    positions = []
    for i in frame_ids:
        T = extrinsics_list[i]
        R = T[:3, :3]
        t = T[:3, 3]
        positions.append(-R.T @ t)
    return np.asarray(positions)
import gtsam

def plot_optimization_median_projection_error(
    window_ids,
    initial_median_proj_errors,
    final_median_proj_errors,
    output_path: Path,
):
    plt.figure(figsize=(12, 5))
    plt.plot(window_ids, initial_median_proj_errors, label="Initial error", color="orange", linewidth=0.5)
    plt.plot(window_ids, final_median_proj_errors, label="Optimized error", color="blue", linewidth=0.5)
    plt.xlabel("Bundle Starting at frame idx")
    plt.ylabel("Median Projection Error [pixels]")
    plt.title("Median Projection Error Before and After Bundle Optimization")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_trajectory_comparison_all(
    pnp_extrinsics,
    bundle_extrinsics,
    pg_lc_extrinsics,
    gt_extrinsics,
    frame_ids,
    output_path: Path,
):
    pos_gt = positions_from_world_to_camera_extrinsics(gt_extrinsics, frame_ids)
    pos_pnp = positions_from_world_to_camera_extrinsics(pnp_extrinsics, frame_ids)
    
    # bundle_extrinsics and pg_lc_extrinsics are lists of 4x4 camera-to-world matrices
    pos_bundle = np.asarray([m[:3, 3] for m in bundle_extrinsics], dtype=float)
    pos_pg_lc = np.asarray([m[:3, 3] for m in pg_lc_extrinsics], dtype=float)

    plt.figure(figsize=(12, 10))
    plt.plot(pos_gt[:, 0], pos_gt[:, 2], label="ground_truth", color="green", linewidth=0.5, linestyle="--")
    plt.plot(pos_pnp[:, 0], pos_pnp[:, 2], label="PnP_trajectory", color="red", alpha=0.7)
    plt.plot(pos_bundle[:, 0], pos_bundle[:, 2], label="Bundle_trajectory", color="orange", alpha=0.7)
    plt.plot(pos_pg_lc[:, 0], pos_pg_lc[:, 2], label="PoseGraph_LC_trajectory", color="blue", alpha=0.7)

    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("Trajectory Comparison")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def compute_absolute_errors(estimated_poses, gt_poses, frame_ids, is_c2w_list=False):
    if is_c2w_list:
        pos_est = np.asarray([m[:3, 3] for m in estimated_poses])
        angles_est = [gtsam.Pose3(m).rotation() for m in estimated_poses]
    else:
        pos_est = positions_from_world_to_camera_extrinsics(estimated_poses, frame_ids)
        angles_est = [gtsam.Pose3(gtsam.Rot3(estimated_poses[i][:3, :3].T), gtsam.Point3()).rotation() for i in frame_ids]
    
    pos_gt = positions_from_world_to_camera_extrinsics(gt_poses, frame_ids)
    angles_gt = [gtsam.Pose3(gtsam.Rot3(gt_poses[i][:3, :3].T), gtsam.Point3()).rotation() for i in frame_ids]

    err_x = pos_est[:, 0] - pos_gt[:, 0]
    err_y = pos_est[:, 1] - pos_gt[:, 1]
    err_z = pos_est[:, 2] - pos_gt[:, 2]
    err_norm = np.linalg.norm(pos_est - pos_gt, axis=1)

    # Angle error: use Rodrigues axis-angle magnitude (norm of logmap = theta)
    err_angle = []
    for i in range(len(frame_ids)):
        r_gt = angles_gt[i]
        r_est = angles_est[i]
        err_angle.append(np.linalg.norm(r_gt.between(r_est).logmap()) * 180.0 / np.pi)
    
    return err_x, err_y, err_z, err_norm, err_angle

def plot_absolute_estimation_error(estimated_poses, gt_poses, frame_ids, title_prefix, output_path: Path, is_values=False):
    # is_values is reused here to mean is_c2w_list since we don't use Values anymore
    err_x, err_y, err_z, err_norm, err_angle = compute_absolute_errors(estimated_poses, gt_poses, frame_ids, is_c2w_list=is_values)

    plt.figure(figsize=(12, 12))
    
    plt.subplot(2, 1, 1)
    plt.plot(frame_ids, err_x, label="x", alpha=0.8)
    plt.plot(frame_ids, err_y, label="y", alpha=0.8)
    plt.plot(frame_ids, err_z, label="z", alpha=0.8)
    plt.plot(frame_ids, err_norm, label="norm", linewidth=0.5)
    plt.xlabel("frame")
    plt.ylabel("Error [m]")
    plt.title(f"{title_prefix} Location Error")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(frame_ids, err_angle, label="Angle Error (deg)", color='red')
    plt.xlabel("frame")
    plt.ylabel("Error [deg]")
    plt.title(f"{title_prefix} Angular Error")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def extract_relative_poses(poses, frame_ids, is_c2w_list=False):
    rel_poses = []
    if is_c2w_list:
        for i in range(len(poses) - 1):
            p1 = gtsam.Pose3(poses[i])
            p2 = gtsam.Pose3(poses[i+1])
            rel_poses.append(p1.between(p2))
    else:
        for i in range(len(frame_ids) - 1):
            T1 = poses[frame_ids[i]]
            T2 = poses[frame_ids[i+1]]
            p1 = gtsam.Pose3(gtsam.Rot3(T1[:3, :3].T), gtsam.Point3(-T1[:3, :3].T @ T1[:3, 3]))
            p2 = gtsam.Pose3(gtsam.Rot3(T2[:3, :3].T), gtsam.Point3(-T2[:3, :3].T @ T2[:3, 3]))
            rel_poses.append(p1.between(p2))
    return rel_poses

def plot_relative_estimation_error(bundle_extrinsics, pnp_extrinsics, gt_extrinsics, frame_ids, output_path: Path):
    rel_gt = extract_relative_poses(gt_extrinsics, frame_ids, False)
    rel_bundle = extract_relative_poses(bundle_extrinsics, frame_ids, True)
    rel_pnp = extract_relative_poses(pnp_extrinsics, frame_ids, False)

    err_bundle_norm = []
    err_bundle_angle = []
    err_pnp_norm = []
    err_pnp_angle = []

    for i in range(len(rel_gt)):
        err_bundle_norm.append(np.linalg.norm(rel_gt[i].translation() - rel_bundle[i].translation()))
        # Rodrigues axis-angle magnitude (norm of logmap = theta)
        err_bundle_angle.append(np.linalg.norm(rel_gt[i].rotation().between(rel_bundle[i].rotation()).logmap()) * 180.0 / np.pi)
        
        err_pnp_norm.append(np.linalg.norm(rel_gt[i].translation() - rel_pnp[i].translation()))
        err_pnp_angle.append(np.linalg.norm(rel_gt[i].rotation().between(rel_pnp[i].rotation()).logmap()) * 180.0 / np.pi)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(frame_ids[:-1], err_bundle_norm, label="Bundle Relative Norm (m)")
    plt.plot(frame_ids[:-1], err_pnp_norm, label="PnP Relative Norm (m)")
    plt.ylabel("Translation Error [m]")
    plt.legend()
    plt.grid(True)
    plt.title("Relative Pose Estimation Error")

    plt.subplot(2, 1, 2)
    plt.plot(frame_ids[:-1], err_bundle_angle, label="Bundle Relative Angle (deg)")
    plt.plot(frame_ids[:-1], err_pnp_angle, label="PnP Relative Angle (deg)")
    plt.xlabel("Frame ID")
    plt.ylabel("Angle Error [deg]")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_relative_error_subsections_line(estimated_poses, gt_poses, subsection_pairs, title_prefix, output_path: Path, is_c2w_list=False):
    """Plot relative estimation error for sequence lengths 100, 400, 800.

    Args:
        subsection_pairs: dict {length: [(i, j, frame_id_i, frame_id_j), ...]}
            Precomputed from caching.load_or_build_subsection_pairs().
            Each pair is the exact keyframe indices/IDs for that sequence length.
    """
    lengths = sorted(subsection_pairs.keys())

    # Precompute cumulative GT arc-length for each keyframe in subsection_pairs
    # We need all unique frame_ids that appear as endpoints
    all_frame_ids = sorted({fid for pairs in subsection_pairs.values() for (_, _, fi, fj) in pairs for fid in (fi, fj)})
    all_frame_ids_set = set(all_frame_ids)

    # Build cumulative distance up to each frame index using consecutive GT steps
    max_fid = max(all_frame_ids) + 1
    # Map frame_id -> cumulative distance from frame 0
    cum_dist_by_fid: dict[int, float] = {}
    running = 0.0
    prev_fid = None
    for fid in range(max_fid):
        if prev_fid is not None:
            T1 = gt_poses[prev_fid]
            T2 = gt_poses[fid]
            p1 = gtsam.Pose3(gtsam.Rot3(T1[:3, :3].T), gtsam.Point3(-T1[:3, :3].T @ T1[:3, 3]))
            p2 = gtsam.Pose3(gtsam.Rot3(T2[:3, :3].T), gtsam.Point3(-T2[:3, :3].T @ T2[:3, 3]))
            running += np.linalg.norm(p1.between(p2).translation())
        if fid in all_frame_ids_set:
            cum_dist_by_fid[fid] = running
        prev_fid = fid

    plt.figure(figsize=(12, 10))

    for plot_idx, metric in enumerate(["Location", "Angle"]):
        plt.subplot(2, 1, plot_idx + 1)
        for length in lengths:
            x_vals = []
            y_vals = []
            for (i, j, frame_id_i, frame_id_j) in subsection_pairs[length]:
                T1_gt = gt_poses[frame_id_i]
                T2_gt = gt_poses[frame_id_j]
                p1_gt = gtsam.Pose3(gtsam.Rot3(T1_gt[:3, :3].T), gtsam.Point3(-T1_gt[:3, :3].T @ T1_gt[:3, 3]))
                p2_gt = gtsam.Pose3(gtsam.Rot3(T2_gt[:3, :3].T), gtsam.Point3(-T2_gt[:3, :3].T @ T2_gt[:3, 3]))
                rel_gt = p1_gt.between(p2_gt)

                if is_c2w_list:
                    p1_est = gtsam.Pose3(estimated_poses[i])
                    p2_est = gtsam.Pose3(estimated_poses[j])
                else:
                    T1_est = estimated_poses[frame_id_i]
                    T2_est = estimated_poses[frame_id_j]
                    p1_est = gtsam.Pose3(gtsam.Rot3(T1_est[:3, :3].T), gtsam.Point3(-T1_est[:3, :3].T @ T1_est[:3, 3]))
                    p2_est = gtsam.Pose3(gtsam.Rot3(T2_est[:3, :3].T), gtsam.Point3(-T2_est[:3, :3].T @ T2_est[:3, 3]))
                rel_est = p1_est.between(p2_est)

                error_pose = rel_est.between(rel_gt)

                total_dist = cum_dist_by_fid[frame_id_j] - cum_dist_by_fid[frame_id_i]
                if total_dist > 0:
                    if metric == "Location":
                        val = (np.linalg.norm(error_pose.translation()) / total_dist) * 100.0
                    else:
                        # Rodrigues axis-angle magnitude (norm of logmap = theta)
                        val = (np.linalg.norm(gtsam.Rot3.Logmap(error_pose.rotation())) * 180.0 / np.pi) / total_dist
                    x_vals.append(frame_id_i)
                    y_vals.append(val)

            plt.plot(x_vals, y_vals, label=f"{length}", linewidth=0.5)

        plt.xlabel("Frame Number")
        plt.ylabel(f"Total {metric} error norm (measure as error%: m/m)" if metric == "Location" else f"{metric} error (measure as deg/m)")
        plt.title(f"Relative {title_prefix} estimation error over sub-sections: {metric}")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Sequence Length")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_loop_closure_stats(lc_stats, matches_out_path: Path, inliers_out_path: Path):
    if not lc_stats:
        return
        
    targets = [s["target_frame"] for s in lc_stats]
    matches = [s["num_matches"] for s in lc_stats]
    inliers = [s["inlier_percentage"] for s in lc_stats]
    
    plt.figure(figsize=(10, 5))
    plt.bar(targets, matches, width=10, color="green", alpha=0.7)
    plt.xlabel("Successful Loop Closure Frame (Target)")
    plt.ylabel("Number of Matches")
    plt.title("Number of Matches per successful loop closure frame")
    plt.grid(alpha=0.3)
    matches_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(matches_out_path, dpi=200)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.bar(targets, inliers, width=10, color="orange", alpha=0.7)
    plt.xlabel("Successful Loop Closure Frame (Target)")
    plt.ylabel("Inlier Percentage [%]")
    plt.title("Inlier percentage per successful loop closure frame")
    plt.grid(alpha=0.3)
    inliers_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(inliers_out_path, dpi=200)
    plt.close()

def safe_log_det(matrix):
    det = np.linalg.det(matrix)
    return np.log10(det) if det > 0 else 0

def plot_uncertainty(versions_data, lc_stats, keyframe_ids, output_path: Path):
    # versions_data contains 'covariances' dict mapping fid -> 6x6 matrix
    # Pose Graph (no LC) is usually versions_data[0]
    # Pose Graph (with LC) is usually versions_data[1]
    if len(versions_data) < 2 or versions_data[0]["covariances"] is None or versions_data[1]["covariances"] is None:
        return
        
    covs_ba = versions_data[0]["covariances"]
    covs_lc = versions_data[1]["covariances"]
    
    lc_targets = [s["target_frame"] for s in lc_stats] if lc_stats else []
    
    plt.figure(figsize=(12, 10))
    # --- Location Uncertainty ---
    det_ba_loc = [safe_log_det(covs_ba[kf][3:6, 3:6]) for kf in keyframe_ids]
    det_lc_loc = [safe_log_det(covs_lc[kf][3:6, 3:6]) for kf in keyframe_ids]
    
    # Shift to start at 0
    det_ba_loc = np.array(det_ba_loc) - det_ba_loc[0]
    det_lc_loc = np.array(det_lc_loc) - det_lc_loc[0]

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(keyframe_ids, det_ba_loc, label="uncertainty score BA", color="dodgerblue", alpha=0.6, linewidth=1.5)
    plt.plot(keyframe_ids, det_lc_loc, label="uncertainty score LC", color="sandybrown", alpha=0.8, linewidth=1.5)
    
    if lc_targets:
        plt.scatter(lc_targets, [0] * len(lc_targets), color="tab:blue", label="Loop Closure Location", zorder=5)
    
    plt.xlabel("KeyFrame Index")
    plt.ylabel("uncertainty score per frame")
    plt.title("Uncertainty size vs keyframe - Location Uncertainty (log10 det, shifted)")
    plt.legend()
    plt.grid(alpha=0.3)

    # --- Angle Uncertainty ---
    det_ba_rot = [safe_log_det(covs_ba[kf][0:3, 0:3]) for kf in keyframe_ids]
    det_lc_rot = [safe_log_det(covs_lc[kf][0:3, 0:3]) for kf in keyframe_ids]
    
    # Shift to start at 0
    det_ba_rot = np.array(det_ba_rot) - det_ba_rot[0]
    det_lc_rot = np.array(det_lc_rot) - det_lc_rot[0]

    plt.subplot(2, 1, 2)
    plt.plot(keyframe_ids, det_ba_rot, label="uncertainty score BA", color="dodgerblue", alpha=0.6, linewidth=1.5)
    plt.plot(keyframe_ids, det_lc_rot, label="uncertainty score LC", color="sandybrown", alpha=0.8, linewidth=1.5)
    
    if lc_targets:
        plt.scatter(lc_targets, [0] * len(lc_targets), color="tab:blue", label="Loop Closure Location", zorder=5)
    plt.title("Uncertainty size vs keyframe - Angle Uncertainty (log10 det)")
    plt.xlabel("KeyFrame Index")
    plt.ylabel("uncertainty score per frame")
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
