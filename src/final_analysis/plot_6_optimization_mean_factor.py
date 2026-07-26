import numpy as np
import matplotlib.pyplot as plt
from src.slam.ba.window_solver import solve_all_bundle_windows
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import pickle
from pathlib import Path
from time import perf_counter
import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import C, Q
from tqdm import tqdm
from src.slam.ba.gtsam_utils import (
    make_stereo_camera,
    pose3_from_world_to_camera_extrinsic,
    stereo_point_from_triplet,
    make_gtsam_stereo_calibration,
    stereo_image_distances,
)
from src.slam.ba.optimization import _validate_graph_keys
from src.slam.ba.results import BundleAdjustmentResult, ProjectionFactorMetadata, BundleWindowSolution
from src.slam.ba.window_selection import collect_window_tracks, bundle_windows_from_keyframes, choose_keyframes_by_motion
from src.slam.config import (
    SEQUENCE_DIR,
    CACHE_DIR,
    GT_POSES_PATH,
    GLOBAL_CAMERA_MATRICES_PATH
)
from src.slam.database.facade import SlamDatabase
from src.slam.features.detectors import extract_features
from src.slam.features.matching import match_and_filter, get_matched_points
from src.slam.geometry.ransac import ransac_pnp
from src.slam.geometry.transforms import to_homogeneous_transform
from src.slam.io.calibration import read_stereo_calibration
from src.slam.io.image_loader import read_images
from src.slam.pipeline.database_pipeline import _create_frame_data, _match_temporal_features
from src.slam.geometry.triangulation import triangulate_opencv
from src.slam.loop_closure.candidate_detection import (
    ConsensusMatcher,
    ConsensusMatchResult,
    LoopClosureCandidate,
    RelativePoseEstimate,
    detect_candidates_for_keyframe,
    refine_relative_pose_with_bundle_adjustment,
    score_candidates_for_keyframe,
    select_spread_loop_closures,
)
from src.slam.pose_graph.constraints import extract_relative_pose_constraint
from src.slam.pose_graph.graph_builder import build_pose_graph
from src.slam.pose_graph.optimization import optimize_pose_graph
from src.slam.visualization.ex7_plots import (
    plot_absolute_location_error,
    plot_consensus_match,
    plot_location_uncertainty,
    plot_pose_graph_comparisons,
    plot_pose_graphs_versions,
    positions_from_values,
)
from src.slam.pipeline.pose_graph_pipeline import solve_bundle_windows_and_extract_constraints
from src.ex8 import build_database, load_or_build_db, load_or_build_pose_graph_with_lc, load_or_build_pose_graph_no_lc, CACHE_DIR

from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_or_build_bundle_windows()["errors"]
    w_ids = data["window_ids"]
    plt.figure(figsize=(10, 4))
    plt.plot(w_ids, data["initial_average_errors"], label="Initial", marker="o", markersize=3)
    plt.plot(w_ids, data["final_average_errors"], label="Optimized", marker="o", markersize=3)
    plt.xlabel("Window ID")
    plt.ylabel("Mean Factor Error")
    plt.title("Optimization Mean Factor Error")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "6_optimization_mean_factor_error.png", dpi=150)
if __name__ == "__main__": main()


def extract_bundle_window_average_errors(bundle_solutions):
    """Extract normalized factor errors from solved BA windows."""
    window_ids = []
    initial_average_errors = []
    final_average_errors = []

    for window_index, solution in enumerate(bundle_solutions, start=1):
        result = solution.result

        if result.num_factors <= 0:
            continue

        initial_error = float(result.average_initial_error)
        final_error = float(result.average_final_error)

        if not np.isfinite(initial_error) or not np.isfinite(final_error):
            continue

        window_ids.append(window_index)
        initial_average_errors.append(initial_error)
        final_average_errors.append(final_error)

    return (
        np.asarray(window_ids, dtype=int),
        np.asarray(initial_average_errors, dtype=float),
        np.asarray(final_average_errors, dtype=float),
    )

def compute_bundle_window_errors(bundle_solutions):
    window_ids = []
    initial_average_errors = []
    final_average_errors = []
    initial_median_proj_errors = []
    final_median_proj_errors = []

    for window_index, solution in enumerate(
        bundle_solutions,
        start=1,
    ):
        result = solution.result

        if result.num_factors <= 0:
            continue

        initial_error = float(
            result.average_initial_error
        )
        final_error = float(
            result.average_final_error
        )

        if (
            not np.isfinite(initial_error)
            or not np.isfinite(final_error)
        ):
            continue

        init_proj_errors = []
        final_proj_errors = []
        for metadata in result.projection_factor_metadata:
            factor = result.graph.at(metadata.factor_index)
            # unwhitenedError returns [uL_err, uR_err, v_err]
            e_init = factor.unwhitenedError(result.initial)
            e_final = factor.unwhitenedError(result.optimized)
            init_proj_errors.append(np.linalg.norm(e_init))
            final_proj_errors.append(np.linalg.norm(e_final))

        window_ids.append(window_index)
        initial_average_errors.append(initial_error)
        final_average_errors.append(final_error)
        initial_median_proj_errors.append(np.median(init_proj_errors) if init_proj_errors else 0.0)
        final_median_proj_errors.append(np.median(final_proj_errors) if final_proj_errors else 0.0)

    return (
        np.asarray(window_ids, dtype=int),
        np.asarray(initial_average_errors, dtype=float),
        np.asarray(final_average_errors, dtype=float),
        np.asarray(initial_median_proj_errors, dtype=float),
        np.asarray(final_median_proj_errors, dtype=float),
    )

@dataclass
class BundleAnalysisContext:
    calibration: gtsam.Cal3_S2Stereo

    keyframes: list[int]

    solutions: list[BundleWindowSolution]

def prepare_bundle_adjustment_analysis(
    database,
) -> BundleAnalysisContext:
    """Select motion-based keyframes and solve all BA windows once."""

    P1, P2 = read_stereo_calibration()

    calibration = make_gtsam_stereo_calibration(P1, P2)

    pnp_extrinsics = np.asarray(
        np.load(GLOBAL_CAMERA_MATRICES_PATH),
        dtype=float,
    )

    keyframes = choose_keyframes_by_motion(
        poses=pnp_extrinsics,
        min_gap=11,
        max_gap=20,
        min_translation=5.0,
        min_rotation_deg=12.0,
        target_translation=5.0,
        target_rotation_deg=12.0,
    )

    print(f"Selected {len(keyframes)} motion-based keyframes.")

    print(f"First keyframes: {keyframes[:10]}")

    print(f"Last keyframes:  {keyframes[-10:]}")

    solutions = solve_all_bundle_windows(
        slam_db=database,
        calibration=calibration,
        keyframes=keyframes,
        min_track_observations=2,
        measurement_sigma_pixels=1.0,
        verbose=True,
    )

    return BundleAnalysisContext(
        calibration=calibration,
        keyframes=keyframes,
        solutions=solutions,
    )

def get_stereo_observation(manager_2d, frame_id, track_id):
    """Return the observation as (x_left, x_right, y).

    This isolates the database-specific API in one place. If manager_2d does
    not expose link_triplet directly, only this function needs to be changed.
    """
    if hasattr(manager_2d, "link_triplet"):
        return manager_2d.link_triplet(frame_id, track_id)

    # Some SlamDatabase facades expose the old TrackingDB through an
    # attribute. Keep this branch only if it matches your implementation.
    if hasattr(manager_2d, "database") and hasattr(
        manager_2d.database,
        "link_triplet",
    ):
        return manager_2d.database.link_triplet(frame_id, track_id)

    raise AttributeError(
        "Could not retrieve a stereo observation. Implement "
        "get_stereo_observation() using the manager_2d link API. "
        "It must return (x_left, x_right, y)."
    )

def compute_bundle_adjustment_analysis(database):
    """Optimize each track landmark and compute BA reprojection errors."""
    print("\n" + "=" * 60)
    print("Track Bundle-Adjustment projection-error analysis")
    print("=" * 60)

    pnp_extrinsics = np.asarray(
        np.load(GLOBAL_CAMERA_MATRICES_PATH),
        dtype=float,
    )

    world_to_camera_by_frame = {
        frame_id: to_homogeneous_transform(extrinsic)
        for frame_id, extrinsic in enumerate(pnp_extrinsics)
    }

    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)

    (
        distances,
        median_left_errors,
        median_right_errors,
        sample_counts,
    ) = compute_median_projection_errors_by_distance(
        database=database,
        world_to_camera_by_frame=world_to_camera_by_frame,
        calibration=calibration,
        optimize_landmarks=True,
        min_track_length=2,
        max_distance=50,
    )

    return {
        "distances": distances,
        "median_left_errors": median_left_errors,
        "median_right_errors": median_right_errors,
        "sample_counts": sample_counts,
    }

def load_or_build_bundle_windows():
    import pickle
    path = CACHE_DIR / "bundle_windows.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
            
    db = load_or_build_db()
    ba_context = prepare_bundle_adjustment_analysis(db)
    
    (
        window_ids,
        initial_average_errors,
        final_average_errors,
        initial_median_proj_errors,
        final_median_proj_errors,
    ) = compute_bundle_window_errors(ba_context.solutions)
    
    errors = {
        "window_ids": window_ids,
        "initial_average_errors": initial_average_errors,
        "final_average_errors": final_average_errors,
        "initial_median_proj_errors": initial_median_proj_errors,
        "final_median_proj_errors": final_median_proj_errors,
    }
    
    ba_context.calibration = None
    data = {
        "context": ba_context,
        "errors": errors
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data
