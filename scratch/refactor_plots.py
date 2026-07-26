import ast
import os
from pathlib import Path

def extract_functions(source_code, func_names):
    tree = ast.parse(source_code)
    extracted = []
    
    # We also need imports? We will just hardcode the necessary imports.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            start = node.lineno - 1
            end = node.end_lineno
            lines = source_code.split('\n')[start:end]
            extracted.append('\n'.join(lines))
        elif isinstance(node, ast.ClassDef) and node.name in func_names:
            # Need to get decorators too
            start = node.decorator_list[0].lineno - 1 if node.decorator_list else node.lineno - 1
            end = node.end_lineno
            lines = source_code.split('\n')[start:end]
            extracted.append('\n'.join(lines))
            
    return '\n\n'.join(extracted)

def main():
    ex8_path = Path("src/ex8.py")
    with open(ex8_path, "r") as f:
        ex8_source = f.read()

    # Get imports
    imports = []
    tree = ast.parse(ex8_source)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno
            lines = ex8_source.split('\n')[start:end]
            imports.append('\n'.join(lines))
    import_code = '\n'.join(imports)
    import_code += "\nfrom src.ex8 import build_database, load_or_build_db, load_or_build_pose_graph_with_lc, load_or_build_pose_graph_no_lc, CACHE_DIR\n"

    # 1. Inlier percentages -> plot_2
    inlier_funcs = ["load_or_build_inlier_percentages"]
    inlier_code = extract_functions(ex8_source, inlier_funcs)
    
    # 2. Tracking analysis -> plot_1, plot_3, plot_4
    tracking_funcs = ["load_or_build_tracking_analysis", "compute_pnp_analysis", "_get_result_value", "get_stereo_observation", "get_relative_camera_transform", "compose_camera_transform", "pose3_camera_to_world_to_extrinsic"]
    tracking_code = extract_functions(ex8_source, tracking_funcs)
    
    # 3. Bundle windows -> plot_6, plot_7, plot_14
    bundle_funcs = ["load_or_build_bundle_windows", "extract_bundle_window_average_errors", "compute_bundle_window_errors", "BundleAnalysisContext", "prepare_bundle_adjustment_analysis", "compute_bundle_adjustment_analysis", "get_stereo_observation"]
    bundle_code = extract_functions(ex8_source, bundle_funcs)
    
    # 4. Projection data -> plot_8, plot_9
    proj_funcs = ["load_or_build_projection_data", "compute_median_projection_errors_by_distance", "optimize_track_landmark", "get_stereo_observation"]
    proj_code = extract_functions(ex8_source, proj_funcs)
    
    # Write to plot 2
    p2 = Path("src/final_analysis/plot_2_inlier_percentage.py")
    with open(p2, "r") as f: content = f.read()
    content = content.replace("from src.ex8 import load_or_build_inlier_percentages", "")
    with open(p2, "w") as f: f.write(content + "\n\n" + inlier_code + "\n")
    
    # Write to plots 1, 3, 4
    for p in ["plot_1_matches.py", "plot_3_connectivity.py", "plot_4_track_length.py"]:
        path = Path("src/final_analysis") / p
        with open(path, "r") as f: content = f.read()
        content = content.replace("from src.ex8 import load_or_build_tracking_analysis", import_code)
        with open(path, "w") as f: f.write(content + "\n\n" + tracking_code + "\n")

    # Write to plots 6, 7, 14
    for p in ["plot_6_optimization_mean_factor.py", "plot_7_optimization_median_projection.py", "plot_14_relative_bundle_subsections.py"]:
        path = Path("src/final_analysis") / p
        with open(path, "r") as f: content = f.read()
        content = content.replace("from src.ex8 import load_or_build_bundle_windows", import_code)
        with open(path, "w") as f: f.write(content + "\n\n" + bundle_code + "\n")
        
    # Write to plots 8, 9
    for p in ["plot_8_pnp_projection.py", "plot_9_bundle_projection.py"]:
        path = Path("src/final_analysis") / p
        with open(path, "r") as f: content = f.read()
        content = content.replace("from src.ex8 import load_or_build_projection_data", import_code)
        with open(path, "w") as f: f.write(content + "\n\n" + proj_code + "\n")

    print("Injected functions into plot scripts.")

if __name__ == "__main__":
    main()
