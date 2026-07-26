import re
from pathlib import Path

def main():
    path = Path("src/ex8.py")
    with open(path, "r") as f:
        lines = f.readlines()
        
    # Functions to delete entirely
    to_delete = [
        "load_or_build_inlier_percentages",
        "load_or_build_tracking_analysis",
        "compute_pnp_analysis",
        "_get_result_value",
        "load_or_build_bundle_windows",
        "extract_bundle_window_average_errors",
        "compute_bundle_window_errors",
        "prepare_bundle_adjustment_analysis",
        "compute_bundle_adjustment_analysis",
        "load_or_build_projection_data",
        "compute_median_projection_errors_by_distance",
        "optimize_track_landmark",
        "get_stereo_observation",
        "get_relative_camera_transform",
        "compose_camera_transform",
        "pose3_camera_to_world_to_extrinsic",
        "_is_geometrically_close",
        "_print_mahalanobis_threshold_sweep",
        "_closest_spatial_candidate",
        "_load_left_image",
        "_show_candidate_pair",
        "_print_candidate_debug",
        "_group_candidates_by_target",
        "q_1",
        "q_2",
        "q_3",
        "q_4",
    ]
    
    # We will identify the start of each top-level definition
    # and if it matches one to delete, we skip until the next top-level definition.
    # Wait, we need to INLINE q_1...q_4, so we must extract their bodies first!
    
    # Actually, the user says "ex8 doesnt need q1,...,q4 these are from prevoius ex we can remove them"
    # Wait, what if they meant they literally don't need them because they're already in `ex7_plots` or `ex7.py` and we don't even need to inline them, we just remove them because the user doesn't care about building the pose graph again?
    # BUT `load_or_build_pose_graph_no_lc()` needs them.
    # If I just inline the bodies of `q_1`, `q_2`, `q_3`, `q_4` into `load_or_build_...` it satisfies "we can remove them" (as function definitions).
    
    # Let's write a parser to extract function bodies
    def extract_body(func_name):
        body_lines = []
        in_func = False
        indent = ""
        for line in lines:
            if line.startswith(f"def {func_name}("):
                in_func = True
                continue
            if in_func:
                if line.startswith("def ") or line.startswith("class ") or (line.strip() and not line.startswith(" ") and not line.startswith("\t") and not line.startswith(")")):
                    if line.startswith(")") and not line.strip() == ")": pass
                    elif line.strip() == ")": continue
                    elif line.strip() == "": continue
                    else:
                        break
                body_lines.append(line)
        # Remove trailing docstrings and remove one level of indentation
        # wait, python indent is 4 spaces
        cleaned = []
        for line in body_lines:
            if line.startswith("    "):
                cleaned.append(line[4:])
            else:
                cleaned.append(line)
        return "".join(cleaned)
        
    q1_body = extract_body("q_1")
    q2_body = extract_body("q_2")
    q3_body = extract_body("q_3")
    q4_body = extract_body("q_4")
    
    new_lines = []
    skip = False
    for line in lines:
        is_top_level = line.startswith("def ") or line.startswith("class ")
        if is_top_level:
            func_name = line.split(" ")[1].split("(")[0]
            if func_name in to_delete:
                skip = True
            elif func_name == "BundleAnalysisContext":
                skip = True
            else:
                skip = False
                
        if not skip:
            new_lines.append(line)
            
    # Now replace the calls to q_1, etc. in load_or_build_...
    content = "".join(new_lines)
    
    # In load_or_build_pose_graph_no_lc:
    # replace `candidates_by_target, build_result, pose_graph_result, slam_db, keyframe_ids = q_1()`
    # with the body of q1
    q1_call = "    candidates_by_target, build_result, pose_graph_result, slam_db, keyframe_ids = q_1()\n"
    q1_inline = "\n".join(["    " + l for l in q1_body.split('\n')]) + "\n"
    # Wait, q_1 returns a tuple, we just change the return to variable assignments
    q1_inline = q1_inline.replace("return (\n        candidates_by_target,\n        build_result,\n        pose_graph_result,\n        slam_db,\n        keyframe_ids,\n    )", "")
    
    content = content.replace(q1_call, q1_inline)
    
    # In load_or_build_loop_closures:
    q23_call = """    selected_results = q_2(pg_no_lc["candidates_by_target"])
    estimates = q_3(selected_results)"""
    q2_inline = "\n".join(["    " + l for l in q2_body.split('\n')]) + "\n"
    q2_inline = q2_inline.replace("return selected_results", "")
    q2_inline = q2_inline.replace("candidates_by_target: dict[int, list[LoopClosureCandidate]],", "candidates_by_target = pg_no_lc[\"candidates_by_target\"]")
    
    q3_inline = "\n".join(["    " + l for l in q3_body.split('\n')]) + "\n"
    q3_inline = q3_inline.replace("return estimates", "")
    
    # We will just manually fix these if this script is too complex, but let's try it.
    
    with open("src/ex8.py", "w") as f:
        f.write(content)
        
if __name__ == "__main__":
    main()
