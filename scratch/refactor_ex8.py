import re

def refactor():
    with open("src/ex8.py", "r") as f:
        content = f.read()
        
    # We will just write a new ex8.py with only the necessary components
    # The necessary components are:
    # 1. Imports
    # 2. build_database
    # 3. load_or_build_db
    # 4. load_or_build_pose_graph_no_lc (with q_1 inlined)
    # 5. load_or_build_loop_closures (with q_2, q_3 inlined)
    # 6. load_or_build_pose_graph_with_lc (with q_4 inlined)
    
    # Actually, the easiest way to inline q_1..q_4 is to just replace the function signatures.
    # But wait, there are returns. Let's just manually construct the new ex8.py because we know exactly what should be in it!
    
    new_ex8 = """import os
from pathlib import Path
import pickle
import numpy as np
import gtsam

from src.slam.config import CACHE_DIR
from src.slam.database.facade import SlamDatabase
from src.slam.pipeline.db_builder import build_database as pipeline_build_db
from src.slam.pipeline.pose_graph_pipeline import (
    solve_bundle_windows_and_extract_constraints,
    build_pose_graph,
    optimize_pose_graph,
from src.slam.loop_closure.candidate_detection import (
    ConsensusMatcher,
    detect_candidates_for_keyframe,
    refine_relative_pose_with_bundle_adjustment,
    score_candidates_for_keyframe,
    select_spread_loop_closures,
)
from src.slam.pose_graph.graph_builder import build_pose_graph
from src.slam.pose_graph.optimization import optimize_pose_graph
from src.slam.pose_graph.constraints import extract_relative_pose_constraint
from src.slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic, positions_from_values

def build_database():
    slam_db, calibration = pipeline_build_db(
        start_frame=0,
        end_frame=3300,
        frames_per_bundle=15,
        overlap_frames=3,
        force_rebuild=False,
    )
    return slam_db, calibration

def load_or_build_db():
    full_db_path = CACHE_DIR / "slam_db.pkl"
    if full_db_path.exists():
        return SlamDatabase.load(str(CACHE_DIR / "slam_db"))
    return build_database()[0]

def load_or_build_pose_graph_no_lc():
    path = CACHE_DIR / "pose_graph_no_lc.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    slam_db, calibration = build_database()
    _, constraints = solve_bundle_windows_and_extract_constraints(
        slam_db=slam_db,
        calibration=calibration,
        verbose=True,
    )
    
    first_frame = constraints[0].start_frame
    first_pose = pose3_from_world_to_camera_extrinsic(slam_db.manager_poses.get_pose(first_frame))
    
    build_result = build_pose_graph(constraints=constraints, first_pose=first_pose)
    pose_graph_result = optimize_pose_graph(
        graph=build_result.graph,
        initial_estimates=build_result.initial_estimates,
        keyframe_ids=build_result.keyframe_ids,
    )
    
    keyframe_ids = build_result.keyframe_ids
    
    candidates_by_target = {}
    for target_frame in keyframe_ids:
        candidates = detect_candidates_for_keyframe(
            optimized_values=pose_graph_result.optimized_estimates,
            covariance_graph=build_result.covariance_graph,
            keyframe_ids=keyframe_ids,
            target_frame=target_frame,
            min_keyframe_separation=5,
            mahalanobis_threshold=32023.8,
        )
        if candidates:
            candidates_by_target[target_frame] = candidates
    
    data = {
        "candidates_by_target": candidates_by_target,
        "build_result": build_result,
        "pose_graph_result": pose_graph_result,
        "keyframe_ids": keyframe_ids
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data

def load_or_build_loop_closures():
    path = CACHE_DIR / "loop_closures.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    pg_no_lc = load_or_build_pose_graph_no_lc()
    
    matcher = ConsensusMatcher(feature_type="akaze", min_loop_inliers=20)
    candidates = [c for target_candidates in pg_no_lc["candidates_by_target"].values() for c in target_candidates]
    
    verified_results = []
    for candidate in candidates:
        res = matcher.verify(candidate)
        if res.success:
            verified_results.append(res)
            
    robust_results = [r for r in verified_results if r.num_inliers >= 30]
    selected_results = select_spread_loop_closures(robust_results, num_representatives=10)
    
    estimates = []
    for result in selected_results:
        estimate = refine_relative_pose_with_bundle_adjustment(result)
        covariance_condition = np.linalg.cond(estimate.covariance)
        if covariance_condition <= 1e8:
            estimates.append(estimate)
            
    lc_stats = []
    for r in selected_results:
        if r.success:
            lc_stats.append({
                "target_frame": r.candidate.target_frame,
                "source_frame": r.candidate.source_frame,
                "num_matches": r.num_four_view_matches,
                "inlier_percentage": r.inlier_ratio * 100.0,
            })
            
    match_data = None
    if selected_results:
        first = selected_results[0]
        match_data = {
            "source_frame": first.candidate.source_frame,
            "target_frame": first.candidate.target_frame,
            "source_left": first.source_left,
            "target_left": first.target_left,
            "source_inliers": first.source_inliers,
            "target_inliers": first.target_inliers,
            "source_outliers": first.source_outliers,
            "target_outliers": first.target_outliers,
        }
        
    data = {
        "estimates": estimates,
        "lc_stats": lc_stats,
        "match_data": match_data,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data

def load_or_build_pose_graph_with_lc():
    path = CACHE_DIR / "pose_graph_with_lc.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    pg_no_lc = load_or_build_pose_graph_no_lc()
    lc = load_or_build_loop_closures()
    
    updated_graph = pg_no_lc["build_result"].graph.clone()
    current_estimates = pg_no_lc["pose_graph_result"].optimized_estimates
    versions = []
    
    marginals_no_lc = gtsam.Marginals(updated_graph, current_estimates)
    versions.append(("Optimized Without LC", current_estimates, marginals_no_lc))
    
    for idx, estimate in enumerate(lc["estimates"]):
        source_key = gtsam.symbol("c", estimate.consensus_result.candidate.source_frame)
        target_key = gtsam.symbol("c", estimate.consensus_result.candidate.target_frame)
        
        noise_model = gtsam.noiseModel.Gaussian.Covariance(estimate.covariance)
        constraint = gtsam.BetweenFactorPose3(
            source_key, target_key, estimate.relative_pose, noise_model
        )
        updated_graph.add(constraint)
        
        opt_res = optimize_pose_graph(
            graph=updated_graph,
            initial_estimates=current_estimates,
            keyframe_ids=pg_no_lc["keyframe_ids"]
        )
        current_estimates = opt_res.optimized_estimates
        
        if idx == len(lc["estimates"]) - 1:
            marginals = gtsam.Marginals(updated_graph, current_estimates)
            versions.append((f"LC {idx+1}", current_estimates, marginals))
    
    updated_pose_graph_result = opt_res
    
    keyframe_ids = pg_no_lc["keyframe_ids"]
    versions_data = []
    for title, values, marginals in versions:
        positions = positions_from_values(values, keyframe_ids)
        covs = {}
        if marginals is not None:
            for fid in keyframe_ids:
                covs[fid] = np.asarray(marginals.marginalCovariance(gtsam.symbol("c", fid)))
        versions_data.append({
            "title": title,
            "positions": positions,
            "covs": covs,
        })
        
    pose_graph_matrices = [
        versions[0][1].atPose3(gtsam.symbol("c", fid)).matrix()
        for fid in keyframe_ids
    ]
    pose_graph_lc_matrices = [
        versions[-1][1].atPose3(gtsam.symbol("c", fid)).matrix()
        for fid in keyframe_ids
    ]
    
    data = {
        "updated_pose_graph_result": updated_pose_graph_result,
        "pose_graph_matrices": pose_graph_matrices,
        "pose_graph_lc_matrices": pose_graph_lc_matrices,
        "versions_data": versions_data,
    }
    
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data
"""

    with open("src/ex8.py", "w") as f:
        f.write(new_ex8)
    print("Refactored ex8.py successfully.")

if __name__ == "__main__":
    refactor()
