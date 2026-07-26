import ast
from pathlib import Path

def get_body_source(source, func_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start = node.body[0].lineno - 1
            end = node.end_lineno
            lines = source.split('\n')[start:end]
            return '\n'.join(lines)
    return ""

def main():
    with open("scratch/ex8_copy.py", "r") as f:
        src = f.read()
        
    imports = []
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno
            imports.append('\n'.join(src.split('\n')[start:end]))
            
    import_block = '\n'.join(imports)
    
    build_db = get_body_source(src, "build_database")
    load_db = get_body_source(src, "load_or_build_db")
    
    q1 = get_body_source(src, "q_1")
    # modify q1 to save to pkl
    q1 = q1.replace(
        "return (\n        candidates_by_target,\n        build_result,\n        pose_graph_result,\n        slam_db,\n        keyframe_ids,\n    )",
        "data = {'candidates_by_target': candidates_by_target, 'build_result': build_result, 'pose_graph_result': pose_graph_result, 'keyframe_ids': keyframe_ids}\n    with open(path, 'wb') as f:\n        pickle.dump(data, f)\n    return data"
    )
    
    q2 = get_body_source(src, "q_2")
    q3 = get_body_source(src, "q_3")
    
    q4 = get_body_source(src, "q_4")
    # modify q4
    q4 = q4.replace(
        "return pose_graph_result, versions",
        "pass"
    )
    
    # We will just write a custom script because we want it to be perfectly formatted.
    
    clean_ex8 = f"""{import_block}

def build_database():
{build_db}

def load_or_build_db():
{load_db}

def load_or_build_pose_graph_no_lc():
    path = CACHE_DIR / "pose_graph_no_lc.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    # From q_1
{q1}

def load_or_build_loop_closures():
    path = CACHE_DIR / "loop_closures.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    pg_no_lc = load_or_build_pose_graph_no_lc()
    candidates_by_target = pg_no_lc["candidates_by_target"]
    
    # From q_2
{q2}

    # From q_3
{q3}

    # Custom stats return
    lc_stats = []
    for r in selected_results:
        if r.success:
            lc_stats.append({{
                "target_frame": r.candidate.target_frame,
                "source_frame": r.candidate.source_frame,
                "num_matches": r.num_four_view_matches,
                "inlier_percentage": r.inlier_ratio * 100.0,
            }})
            
    match_data = None
    successful = [r for r in selected_results if r.success]
    if successful:
        first = successful[0]
        match_data = {{
            "source_frame": first.candidate.source_frame,
            "target_frame": first.candidate.target_frame,
            "source_left": first.source_left,
            "target_left": first.target_left,
            "source_inliers": first.source_inliers,
            "target_inliers": first.target_inliers,
            "source_outliers": first.source_outliers,
            "target_outliers": first.target_outliers,
        }}
        
    data = {{
        "estimates": estimates,
        "lc_stats": lc_stats,
        "match_data": match_data,
    }}
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
    
    graph = pg_no_lc["build_result"].graph
    initial_estimates = pg_no_lc["pose_graph_result"].optimized_estimates
    keyframe_ids = pg_no_lc["keyframe_ids"]
    estimates = lc["estimates"]
    
    # From q_4
{q4}

    versions_data = []
    for title, values, marginals in versions:
        positions = positions_from_values(values, keyframe_ids)
        covs = {{}}
        if marginals is not None:
            for fid in keyframe_ids:
                covs[fid] = np.asarray(marginals.marginalCovariance(gtsam.symbol("c", fid)))
        versions_data.append({{
            "title": title,
            "positions": positions,
            "covs": covs,
        }})
        
    pose_graph_matrices = [
        versions[0][1].atPose3(gtsam.symbol("c", fid)).matrix()
        for fid in keyframe_ids
    ]
    pose_graph_lc_matrices = [
        versions[-1][1].atPose3(gtsam.symbol("c", fid)).matrix()
        for fid in keyframe_ids
    ]
    
    data = {{
        "updated_pose_graph_result": pose_graph_result,
        "pose_graph_matrices": pose_graph_matrices,
        "pose_graph_lc_matrices": pose_graph_lc_matrices,
        "versions_data": versions_data,
    }}
    
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data
"""

    with open("src/ex8.py", "w") as f:
        f.write(clean_ex8)

if __name__ == "__main__":
    main()
