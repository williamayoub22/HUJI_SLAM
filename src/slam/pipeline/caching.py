import pickle
import numpy as np
import gtsam
from gtsam.symbol_shorthand import C
from pathlib import Path
from src.slam import config
from src.slam.database.facade import SlamDatabase
from src.slam.pipeline.database_pipeline import build_database
from src.slam.analysis.bundle_diagnostics import prepare_bundle_adjustment_analysis, compute_bundle_window_errors
from src.slam.pipeline.pose_graph_pipeline import _build_pg_no_lc, _build_pg_with_lc
from src.slam.pipeline.loop_closure_pipeline import _build_lc_candidates, _refine_lc
from src.slam.analysis.track_reprojection import compute_pnp_analysis, compute_bundle_adjustment_analysis


def load_or_build_db():
    full_db_path = config.CACHE_DIR / 'slam_db.pkl'
    if full_db_path.exists():
        return SlamDatabase.load(str(config.CACHE_DIR / 'slam_db'))
    return build_database()[0]

def load_or_build_bundle_windows():
    import pickle
    path = config.CACHE_DIR / 'bundle_windows.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    db = load_or_build_db()
    ba_context = prepare_bundle_adjustment_analysis(db)
    window_ids, initial_average_errors, final_average_errors, initial_median_proj_errors, final_median_proj_errors = compute_bundle_window_errors(ba_context.solutions)
    errors = {'window_ids': window_ids, 'initial_average_errors': initial_average_errors, 'final_average_errors': final_average_errors, 'initial_median_proj_errors': initial_median_proj_errors, 'final_median_proj_errors': final_median_proj_errors}
    ba_context.calibration = None
    data = {'context': ba_context, 'errors': errors}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data

def load_or_build_inlier_percentages():
    import numpy as np
    from src.slam.config import PROJECT_DIR
    import src.slam.config as config
    inlier_path = PROJECT_DIR / 'outputs' / 'ex4' / 'inlier_percentages.npy'
    if not inlier_path.exists():
        total_frames = len(list((config.SEQUENCE_DIR / 'image_0').glob('*.png')))
        return np.ones(total_frames) * config.MAX_PROJECTION_ERROR_DEFAULT
    return np.load(inlier_path)

def load_or_build_pose_graph_no_lc():
    import pickle
    path = config.CACHE_DIR / 'pose_graph_no_lc.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    candidates_by_target, build_result, pose_graph_result, slam_db, keyframe_ids = _build_pg_no_lc()
    data = {'candidates_by_target': candidates_by_target, 'build_result': build_result, 'pose_graph_result': pose_graph_result, 'keyframe_ids': keyframe_ids}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data

def load_or_build_loop_closures():
    import pickle
    path = config.CACHE_DIR / 'loop_closures.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    pg_no_lc = load_or_build_pose_graph_no_lc()
    selected_results = _build_lc_candidates(pg_no_lc['candidates_by_target'])
    estimates = _refine_lc(selected_results)
    lc_stats = []
    for r in selected_results:
        if r.success:
            lc_stats.append({'target_frame': r.candidate.target_frame, 'source_frame': r.candidate.source_frame, 'num_matches': r.num_four_view_matches, 'inlier_percentage': r.inlier_ratio * 100.0})
    match_data = None
    successful = [r for r in selected_results if r.success]
    if successful:
        first = successful[0]
        match_data = {'source_frame': first.candidate.source_frame, 'target_frame': first.candidate.target_frame, 'source_left': first.source_left, 'target_left': first.target_left, 'inlier_mask': first.inlier_mask}
    data = {'selected_results': selected_results, 'estimates': estimates, 'lc_stats': lc_stats, 'match_data': match_data}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data

def load_or_build_pose_graph_with_lc():
    import pickle
    import numpy as np
    path = config.CACHE_DIR / 'pose_graph_with_lc.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    pg_no_lc = load_or_build_pose_graph_no_lc()
    lc = load_or_build_loop_closures()
    updated_pose_graph_result, versions = _build_pg_with_lc(graph=pg_no_lc['build_result'].graph, initial_estimates=pg_no_lc['pose_graph_result'].optimized_estimates, keyframe_ids=pg_no_lc['keyframe_ids'], estimates=lc['estimates'])
    keyframe_ids = pg_no_lc['keyframe_ids']
    versions_data = []
    for title, values, marginals in versions:
        positions = np.array([values.atPose3(C(frame_id)).translation() for frame_id in keyframe_ids if values.exists(C(frame_id))])
        covs = {}
        if marginals is not None:
            for fid in keyframe_ids:
                covs[fid] = np.asarray(marginals.marginalCovariance(gtsam.symbol('c', fid)))
        versions_data.append({'title': title, 'positions': positions, 'covariances': covs if marginals is not None else None})
    pose_graph_matrices = [np.asarray(pg_no_lc['pose_graph_result'].optimized_estimates.atPose3(C(fid)).matrix()) for fid in keyframe_ids]
    pose_graph_lc_matrices = [np.asarray(updated_pose_graph_result.optimized_estimates.atPose3(C(fid)).matrix()) for fid in keyframe_ids]
    data = {'updated_pose_graph_result': updated_pose_graph_result, 'versions_data': versions_data, 'pose_graph_matrices': pose_graph_matrices, 'pose_graph_lc_matrices': pose_graph_lc_matrices}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data

def load_or_build_projection_data():
    import pickle
    path = config.CACHE_DIR / 'projection_data.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    db = load_or_build_db()
    pnp_projection_data = compute_pnp_analysis(db)
    ba_projection_data = compute_bundle_adjustment_analysis(db)
    data = {'pnp_projection_data': pnp_projection_data, 'ba_projection_data': ba_projection_data}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data