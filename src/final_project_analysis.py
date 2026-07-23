from pathlib import Path
import matplotlib.pyplot as plt

from src.slam.ba.results import BundleWindowSolution
from src.slam.ba.window_selection import choose_keyframes_by_motion
from src.slam.config import CACHE_DIR, EX8_OUTPUT_DIR
from src.slam.database.facade import SlamDatabase

from gtsam.symbol_shorthand import C, Q

from slam.ba.gtsam_utils import (
    pose3_from_world_to_camera_extrinsic,
)


from slam.ba.window_solver import solve_all_bundle_windows
from src.slam.config import GLOBAL_CAMERA_MATRICES_PATH

from collections import defaultdict

import gtsam
import numpy as np

from slam.ba.gtsam_utils import (
    make_gtsam_stereo_calibration,
    make_stereo_camera,
    stereo_image_distances,
    stereo_point_from_triplet,
)
from slam.geometry.transforms import to_homogeneous_transform
from slam.io.calibration import read_stereo_calibration
from src.slam.pipeline.pose_graph_pipeline import solve_bundle_windows_and_extract_constraints

DATABASE_PATH = CACHE_DIR / "slam_db"
OUTPUT_DIR = EX8_OUTPUT_DIR
from src.slam.config import PROJECT_DIR

INLIER_PERCENTAGES_PATH = PROJECT_DIR / "outputs" / "ex4" / "inlier_percentages.npy"

from dataclasses import dataclass


def build_or_load_database():
    """Load the tracking database created by the Exercise 8 pipeline."""
    database_file = Path(str(DATABASE_PATH) + ".pkl")

    # todo: change to load or rebuild
    if not database_file.exists():
        raise FileNotFoundError(
            f"Tracking database was not found at {database_file}. "
            "Run the database-building pipeline first."
        )

    print(f"Loading tracking database from {database_file}...")
    return SlamDatabase.load(str(DATABASE_PATH))


def optimize_track_landmark(
    manager_2d,
    track_id,
    frame_ids,
    world_to_camera_by_frame,
    calibration,
    measurement_sigma_pixels=1.0,
):
    """Optimize one landmark using all stereo observations in one track.

    Camera poses are fixed using tight Pose3 priors. Only the landmark is
    meaningfully adjusted by the optimization.
    """
    if len(frame_ids) < 2:
        raise ValueError("At least two observations are required.")

    triangulation_frame = frame_ids[-1]

    triangulation_extrinsic = to_homogeneous_transform(
        world_to_camera_by_frame[triangulation_frame]
    )

    triangulation_camera = make_stereo_camera(
        triangulation_extrinsic,
        calibration,
    )

    triangulation_measurement = stereo_point_from_triplet(
        get_stereo_observation(
            manager_2d,
            triangulation_frame,
            track_id,
        )
    )

    initial_landmark = triangulation_camera.backproject(triangulation_measurement)

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    landmark_key = Q(int(track_id))

    initial_values.insert(
        landmark_key,
        gtsam.Point3(np.asarray(initial_landmark, dtype=float)),
    )

    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        3,
        measurement_sigma_pixels,
    )

    # Very tight priors keep all PnP poses fixed.
    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-7, dtype=float))

    for frame_id in frame_ids:
        pose_key = C(int(frame_id))

        world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[frame_id])

        camera_pose = pose3_from_world_to_camera_extrinsic(world_to_camera)

        initial_values.insert(pose_key, camera_pose)

        graph.add(
            gtsam.PriorFactorPose3(
                pose_key,
                camera_pose,
                pose_prior_noise,
            )
        )

        measurement = stereo_point_from_triplet(
            get_stereo_observation(
                manager_2d,
                frame_id,
                track_id,
            )
        )

        graph.add(
            gtsam.GenericStereoFactor3D(
                measurement,
                measurement_noise,
                pose_key,
                landmark_key,
                calibration,
            )
        )

    optimizer = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_values,
    )

    optimized_values = optimizer.optimize()

    optimized_landmark = np.asarray(
        optimized_values.atPoint3(landmark_key),
        dtype=float,
    ).reshape(3)

    return optimized_landmark


def plot_frame_connectivity(
    frame_ids,
    frame_connectivities,
    output_dir=OUTPUT_DIR,
):
    """Plot the number of tracks shared by each pair of consecutive frames."""
    print("\n" + "=" * 60)
    print("Connectivity")
    print("=" * 60)

    if not frame_connectivities:
        print("WARNING: At least two frames are required to plot connectivity.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Each connectivity value belongs to the transition
    # frame_ids[i] -> frame_ids[i + 1].
    source_frames = frame_ids[:-1]
    mean_connectivity = float(np.mean(frame_connectivities))

    plt.figure(figsize=(12, 4))
    plt.plot(
        source_frames,
        frame_connectivities,
        linewidth=0.7,
        label="Frame",
    )
    plt.axhline(
        y=mean_connectivity,
        linestyle="--",
        linewidth=1.0,
        label=f"Mean = {mean_connectivity:.2f}",
    )

    plt.xlabel("Frame")
    plt.ylabel("Outgoing tracks")
    plt.title("Connectivity")
    plt.xlim(source_frames[0], source_frames[-1])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "frame_connectivity.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Connectivity plot saved to {output_path}")
    return output_path


def plot_track_length_histogram(
    track_lengths,
    output_dir=OUTPUT_DIR,
):
    """Plot the distribution of track lengths on a logarithmic y-axis."""
    print("\n" + "=" * 60)
    print("Track length histogram")
    print("=" * 60)

    if track_lengths.size == 0:
        print("WARNING: No tracks are available for the histogram.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    integer_track_lengths = track_lengths.astype(int)

    # We display only lengths up to 50 because very long tracks are rare
    # and are less informative for visualizing the main distribution.
    max_display_length = 50
    bins = np.arange(0, max_display_length + 2)

    plt.figure(figsize=(12, 4))
    plt.hist(
        integer_track_lengths,
        bins=bins,
        log=True,
        edgecolor="black",
        linewidth=0.3,
    )

    plt.xlim(0, max_display_length)
    plt.xlabel("Track length histogram")
    plt.ylabel("# Tracks [logarithmic scale]")
    plt.title("Track Length Distribution")
    # plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    output_path = output_dir / "track_length_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Track-length histogram saved to {output_path}")
    return output_path


def plot_matches_per_frame(
    frame_ids,
    matches_per_frame,
    output_dir=OUTPUT_DIR,
):
    """Plot the number of feature matches stored in each frame."""
    print("\n" + "=" * 60)
    print("Matches per frame")
    print("=" * 60)

    if not frame_ids:
        print("WARNING: No frames are available.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    mean_matches = float(np.mean(matches_per_frame))

    plt.figure(figsize=(12, 4))
    plt.plot(
        frame_ids,
        matches_per_frame,
        linewidth=0.7,
        label="Matches",
    )
    plt.axhline(
        mean_matches,
        linestyle="--",
        linewidth=1.0,
        label=f"Mean = {mean_matches:.2f}",
    )

    plt.xlim(frame_ids[0], frame_ids[-1])
    plt.xlabel("Frame")
    plt.ylabel("Number of matches")
    plt.title("Number of Matches per Frame")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "matches_per_frame.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Matches-per-frame plot saved to {output_path}")
    return output_path


def compute_median_projection_errors_by_distance(
    database,
    world_to_camera_by_frame,
    calibration,
    *,
    optimize_landmarks,
    min_track_length=2,
    max_distance=50,
):
    """
    Compute median stereo reprojection errors by temporal distance.

    PnP mode:
        1. Backproject the latest stereo observation into the latest camera's
           coordinate system.
        2. Propagate that 3D point into each earlier camera by composing the
           consecutive estimated camera transformations.
        3. Project the propagated point into that frame.
        4. Compare the predicted pixels with the measured stereo observation.

    Bundle-adjustment mode:
        Optimize one world-space landmark using all selected observations,
        while keeping the supplied camera poses fixed.
    """
    manager_2d = database.manager_2d

    left_errors_by_distance = defaultdict(list)
    right_errors_by_distance = defaultdict(list)

    used_tracks = 0
    rejected_tracks = 0

    # An identity-pose camera projects points that are already expressed
    # in the current camera coordinate system.
    local_stereo_camera = make_stereo_camera(
        np.eye(4, dtype=float),
        calibration,
    )

    for track_id in manager_2d.all_tracks():
        track_frames = sorted(manager_2d.frames(track_id))

        usable_frames = [
            frame_id for frame_id in track_frames if frame_id in world_to_camera_by_frame
        ]

        if len(usable_frames) < min_track_length:
            continue

        triangulation_frame = usable_frames[-1]

        evaluation_frames = [
            frame_id
            for frame_id in usable_frames
            if 0 <= triangulation_frame - frame_id <= max_distance
        ]

        if len(evaluation_frames) < min_track_length:
            continue

        try:
            if optimize_landmarks:
                landmark_world = optimize_track_landmark(
                    manager_2d=manager_2d,
                    track_id=track_id,
                    frame_ids=evaluation_frames,
                    world_to_camera_by_frame=world_to_camera_by_frame,
                    calibration=calibration,
                )

                landmark_world = np.asarray(
                    landmark_world,
                    dtype=float,
                ).reshape(3)

                if not np.all(np.isfinite(landmark_world)):
                    raise ValueError("Optimized landmark contains non-finite values.")

            else:
                # Backproject using an identity camera. The result is expressed
                # in triangulation_frame camera coordinates, not world space.
                triangulation_measurement = stereo_point_from_triplet(
                    get_stereo_observation(
                        manager_2d,
                        triangulation_frame,
                        track_id,
                    )
                )

                landmark_in_triangulation_camera = np.asarray(
                    local_stereo_camera.backproject(triangulation_measurement),
                    dtype=float,
                ).reshape(3)

                if not np.all(np.isfinite(landmark_in_triangulation_camera)):
                    raise ValueError("Triangulated landmark contains non-finite values.")

        except (
            KeyError,
            IndexError,
            ValueError,
            RuntimeError,
            gtsam.CheiralityException,
        ):
            rejected_tracks += 1
            continue

        track_contributed = False

        for frame_id in evaluation_frames:
            distance = triangulation_frame - frame_id

            try:
                if optimize_landmarks:
                    frame_extrinsic = to_homogeneous_transform(world_to_camera_by_frame[frame_id])

                    frame_camera = make_stereo_camera(
                        frame_extrinsic,
                        calibration,
                    )

                    projection = frame_camera.project(gtsam.Point3(landmark_world))

                else:
                    # Map the point from the triangulation camera to the
                    # requested evaluation camera by composing estimated
                    # consecutive camera motions.
                    frame_from_triangulation = compose_camera_transform(
                        world_to_camera_by_frame=(world_to_camera_by_frame),
                        source_frame=triangulation_frame,
                        target_frame=frame_id,
                    )

                    landmark_homogeneous = np.append(
                        landmark_in_triangulation_camera,
                        1.0,
                    )

                    landmark_in_frame_camera = (frame_from_triangulation @ landmark_homogeneous)[:3]

                    if not np.all(np.isfinite(landmark_in_frame_camera)):
                        continue

                    # The point is already in frame_id camera coordinates.
                    projection = local_stereo_camera.project(gtsam.Point3(landmark_in_frame_camera))

                measurement = stereo_point_from_triplet(
                    get_stereo_observation(
                        manager_2d,
                        frame_id,
                        track_id,
                    )
                )

                left_error, right_error = stereo_image_distances(
                    measurement=measurement,
                    projection=projection,
                )

            except (
                KeyError,
                IndexError,
                ValueError,
                RuntimeError,
                gtsam.CheiralityException,
            ):
                continue

            left_errors_by_distance[distance].append(float(left_error))
            right_errors_by_distance[distance].append(float(right_error))

            track_contributed = True

        if track_contributed:
            used_tracks += 1
        else:
            rejected_tracks += 1

    distances = np.asarray(
        sorted(set(left_errors_by_distance) & set(right_errors_by_distance)),
        dtype=int,
    )

    median_left_errors = np.asarray(
        [np.median(left_errors_by_distance[distance]) for distance in distances],
        dtype=float,
    )

    median_right_errors = np.asarray(
        [np.median(right_errors_by_distance[distance]) for distance in distances],
        dtype=float,
    )

    sample_counts = np.asarray(
        [
            min(
                len(left_errors_by_distance[distance]),
                len(right_errors_by_distance[distance]),
            )
            for distance in distances
        ],
        dtype=int,
    )

    print(f"Used tracks:        {used_tracks}")
    print(f"Rejected tracks:    {rejected_tracks}")
    print(f"Computed distances: {len(distances)}")

    return (
        distances,
        median_left_errors,
        median_right_errors,
        sample_counts,
    )


def plot_inlier_percentages(
    inlier_percentages,
    output_dir=OUTPUT_DIR,
):
    """Plot the PnP inlier percentage for every frame transition."""
    print("\n" + "=" * 60)
    print("Percentage of inliers per frame")
    print("=" * 60)

    if inlier_percentages is None or len(inlier_percentages) == 0:
        print(
            "WARNING: No inlier-percentage data are available. "
            "Rebuild the Exercise 4 database first."
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    percentages = np.asarray(inlier_percentages, dtype=float).reshape(-1)

    # Support values saved either as ratios [0, 1] or percentages [0, 100].
    if np.nanmax(percentages) <= 1.0:
        percentages = 100.0 * percentages

    # Value i describes the transition from frame i to frame i + 1.
    destination_frames = np.arange(1, len(percentages) + 1)
    mean_percentage = float(np.nanmean(percentages))

    plt.figure(figsize=(12, 4))
    plt.plot(
        destination_frames,
        percentages,
        linewidth=0.7,
        label="PnP inliers",
    )
    plt.axhline(
        y=mean_percentage,
        linestyle="--",
        linewidth=1.0,
        label=f"Mean = {mean_percentage:.2f}%",
    )

    plt.xlim(destination_frames[0], destination_frames[-1])
    plt.ylim(0, 100)
    plt.xlabel("Frame")
    plt.ylabel("Inliers [%]")
    plt.title("Percentage of PnP Inliers per Frame")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "inlier_percentage_per_frame.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Inlier-percentage plot saved to {output_path}")
    return output_path


def _get_result_value(result, possible_names):
    """Read a value from either a dictionary or a result object."""
    for name in possible_names:
        if isinstance(result, dict) and name in result:
            return result[name]

        if hasattr(result, name):
            return getattr(result, name)

    raise AttributeError(f"Could not find any of {possible_names} in bundle-window result.")


def compute_bundle_window_factor_errors(bundle_results):
    """Compute mean factor error before and after BA for each window.

    Each result must contain:
        - the window factor graph;
        - the initial Values;
        - the optimized Values.

    The function supports several common field names so it can work with
    either dictionaries or result objects.
    """
    window_ids = []
    initial_mean_errors = []
    optimized_mean_errors = []

    for window_index, result in enumerate(bundle_results):
        try:
            graph = _get_result_value(
                result,
                (
                    "graph",
                    "factor_graph",
                    "bundle_graph",
                ),
            )

            initial_values = _get_result_value(
                result,
                (
                    "initial_values",
                    "initial",
                    "initial_estimate",
                ),
            )

            optimized_values = _get_result_value(
                result,
                (
                    "optimized_values",
                    "result_values",
                    "optimized",
                    "solution",
                ),
            )

            number_of_factors = int(graph.size())

            if number_of_factors == 0:
                continue

            initial_total_error = float(graph.error(initial_values))
            optimized_total_error = float(graph.error(optimized_values))

            window_ids.append(window_index)
            initial_mean_errors.append(initial_total_error / number_of_factors)
            optimized_mean_errors.append(optimized_total_error / number_of_factors)

        except (
            AttributeError,
            KeyError,
            ValueError,
            RuntimeError,
        ) as error:
            print(f"WARNING: Could not compute errors for bundle window {window_index}: {error}")

    return (
        np.asarray(window_ids, dtype=int),
        np.asarray(initial_mean_errors, dtype=float),
        np.asarray(optimized_mean_errors, dtype=float),
    )


def plot_bundle_window_factor_errors(
    window_ids,
    initial_mean_errors,
    optimized_mean_errors,
    output_dir=OUTPUT_DIR,
):
    """Plot mean factor error before and after BA for every window."""
    print("\n" + "=" * 60)
    print("Bundle-window factor errors")
    print("=" * 60)

    if len(window_ids) == 0:
        print("WARNING: No bundle-window factor errors were computed.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.plot(
        window_ids,
        initial_mean_errors,
        marker="o",
        markersize=3,
        linewidth=1.1,
        label="Initial error",
    )

    plt.plot(
        window_ids,
        optimized_mean_errors,
        marker="o",
        markersize=3,
        linewidth=1.1,
        label="After optimization",
    )

    plt.xlabel("Bundle window")
    plt.ylabel("Mean factor error")
    plt.title("Mean Factor Error Before and After Bundle Adjustment")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "bundle_window_factor_errors.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    mean_initial = float(np.mean(initial_mean_errors))
    mean_optimized = float(np.mean(optimized_mean_errors))

    print(f"Mean initial factor error:   {mean_initial:.6f}")
    print(f"Mean optimized factor error: {mean_optimized:.6f}")

    if mean_initial > 0:
        reduction_percentage = 100.0 * (mean_initial - mean_optimized) / mean_initial

        print(f"Mean error reduction:        {reduction_percentage:.2f}%")

    print(f"Bundle-window error plot saved to {output_path}")

    return output_path


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


def plot_bundle_window_average_errors(
    bundle_solutions,
    output_dir=OUTPUT_DIR,
):
    """Plot normalized factor error before and after BA."""
    window_ids = []
    initial_average_errors = []
    final_average_errors = []

    for window_index, solution in enumerate(
        bundle_solutions,
        start=1,
    ):
        result = solution.result

        if result.num_factors <= 0:
            continue

        # These values are already normalized by the factor count.
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

        window_ids.append(window_index)
        initial_average_errors.append(initial_error)
        final_average_errors.append(final_error)

    if not window_ids:
        print(
            "WARNING: No valid bundle-window errors "
            "were available."
        )
        return None

    window_ids = np.asarray(
        window_ids,
        dtype=int,
    )
    initial_average_errors = np.asarray(
        initial_average_errors,
        dtype=float,
    )
    final_average_errors = np.asarray(
        final_average_errors,
        dtype=float,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        window_ids,
        initial_average_errors,
        linewidth=1.0,
        label="Before optimization",
    )

    plt.plot(
        window_ids,
        final_average_errors,
        linewidth=1.0,
        label="After optimization",
    )

    plt.xlabel("Bundle window")
    plt.ylabel("Average factor error")
    plt.title(
        "Average Factor Error Before and After "
        "Local Bundle Adjustment"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = (
        output_dir
        / "bundle_window_average_factor_error.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    mean_initial = float(
        np.mean(initial_average_errors)
    )
    mean_final = float(
        np.mean(final_average_errors)
    )

    print(
        f"Mean initial average factor error: "
        f"{mean_initial:.6f}"
    )
    print(
        f"Mean final average factor error:   "
        f"{mean_final:.6f}"
    )

    if mean_initial > 0:
        reduction = (
            100.0
            * (mean_initial - mean_final)
            / mean_initial
        )

        print(
            f"Mean error reduction:              "
            f"{reduction:.2f}%"
        )

    print(f"Plot saved to {output_path}")

    return output_path


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


def load_inlier_percentages():
    """Load the per-transition percentages generated in Exercise 4."""
    if not INLIER_PERCENTAGES_PATH.exists():
        print(
            f"WARNING: Inlier percentages were not found at "
            f"{INLIER_PERCENTAGES_PATH}.\n"
            "Run Exercise 4 with REBUILD_DB = True to generate the file."
        )
        return np.asarray([], dtype=float)

    percentages = np.asarray(
        np.load(INLIER_PERCENTAGES_PATH),
        dtype=float,
    ).reshape(-1)

    print(f"Loaded {len(percentages)} inlier percentages from {INLIER_PERCENTAGES_PATH}")

    return percentages


def run_bundle_window_error_analysis(database):
    """Compare initial and optimized factor errors for motion-based BA windows."""
    print("\n" + "=" * 60)
    print("Bundle-window optimization analysis")
    print("=" * 60)

    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)

    # Use the updated motion-based keyframe-selection rule from the
    # pose-graph pipeline. This avoids the old fixed-interval selector.
    _, constraints = solve_bundle_windows_and_extract_constraints(
        slam_db=database,
        calibration=calibration,
        verbose=True,
    )

    if not constraints:
        print("WARNING: No bundle-window constraints were generated.")
        return {
            "window_ids": np.asarray([], dtype=int),
            "initial_mean_errors": np.asarray([], dtype=float),
            "optimized_mean_errors": np.asarray([], dtype=float),
            "plot_path": None,
        }

    # Recover the ordered keyframe sequence from the solved windows.
    keyframes = [constraints[0].start_frame]

    for constraint in constraints:
        if keyframes[-1] != constraint.start_frame:
            raise ValueError(
                "Bundle-window constraints are not consecutive: "
                f"expected start frame {keyframes[-1]}, "
                f"received {constraint.start_frame}."
            )

        keyframes.append(constraint.end_frame)

    print(f"Motion-based keyframes: {len(keyframes)}")
    print(f"Bundle windows:         {len(keyframes) - 1}")

    # Solve the same windows again while retaining the complete solutions
    # needed to evaluate the initial and optimized factor-graph errors.
    bundle_solutions = solve_all_bundle_windows(
        slam_db=database,
        calibration=calibration,
        keyframes=keyframes,
        min_track_observations=2,
        measurement_sigma_pixels=1.0,
        verbose=False,
    )

    (
        window_ids,
        initial_mean_errors,
        optimized_mean_errors,
    ) = compute_bundle_window_factor_errors(bundle_solutions)

    plot_path = plot_bundle_window_factor_errors(
        window_ids=window_ids,
        initial_mean_errors=initial_mean_errors,
        optimized_mean_errors=optimized_mean_errors,
    )

    return {
        "keyframes": keyframes,
        "window_ids": window_ids,
        "initial_mean_errors": initial_mean_errors,
        "optimized_mean_errors": optimized_mean_errors,
        "plot_path": plot_path,
    }


def run_tracking_analysis(database, inlier_percentages):
    """Compute and plot the tracking-performance statistics."""
    manager_2d = database.manager_2d

    frame_ids = sorted(manager_2d.all_frames())
    track_ids = list(manager_2d.all_tracks())

    track_lengths = np.asarray(
        [len(manager_2d.frames(track_id)) for track_id in track_ids],
        dtype=float,
    )

    # Number of stored feature observations/matches in every frame.
    matches_per_frame = [len(manager_2d.tracks(frame_id)) for frame_id in frame_ids]

    # Number of tracks shared by each pair of consecutive database frames.
    frame_connectivities = []

    for current_frame, next_frame in zip(frame_ids[:-1], frame_ids[1:]):
        current_tracks = set(manager_2d.tracks(current_frame))
        next_tracks = set(manager_2d.tracks(next_frame))

        frame_connectivities.append(len(current_tracks.intersection(next_tracks)))

    connectivity_plot_path = plot_frame_connectivity(
        frame_ids,
        frame_connectivities,
    )

    histogram_plot_path = plot_track_length_histogram(
        track_lengths,
    )
    #
    # matches_plot_path = plot_matches_per_frame(
    #     frame_ids,
    #     matches_per_frame,
    # )

    inliers_plot_path = plot_inlier_percentages(
        inlier_percentages,
    )

    results = {
        "number_of_frames": len(frame_ids),
        "number_of_tracks": len(track_ids),
        "mean_track_length": (float(np.mean(track_lengths)) if track_lengths.size > 0 else 0.0),
        "mean_frame_connectivity": (
            float(np.mean(frame_connectivities)) if frame_connectivities else 0.0
        ),
        "mean_matches_per_frame": (float(np.mean(matches_per_frame)) if matches_per_frame else 0.0),
        "mean_inlier_percentage": (
            float(
                np.mean(
                    100.0 * inlier_percentages
                    if (len(inlier_percentages) > 0 and np.max(inlier_percentages) <= 1.0)
                    else inlier_percentages
                )
            )
            if len(inlier_percentages) > 0
            else 0.0
        ),
        "frame_connectivities": frame_connectivities,
        "matches_per_frame": matches_per_frame,
        "connectivity_plot_path": (
            str(connectivity_plot_path) if connectivity_plot_path is not None else None
        ),
        "track_length_histogram_path": (
            str(histogram_plot_path) if histogram_plot_path is not None else None
        ),
        # "matches_per_frame_plot_path": (
        #     str(matches_plot_path)
        #     if matches_plot_path is not None
        #     else None
        # ),
        "inliers_per_frame_plot_path": (
            str(inliers_plot_path) if inliers_plot_path is not None else None
        ),
    }

    print("\n" + "=" * 60)
    print("Tracking database statistics")
    print("=" * 60)
    print(f"Number of frames:          {results['number_of_frames']}")
    print(f"Number of tracks:          {results['number_of_tracks']}")
    print(f"Mean track length:         {results['mean_track_length']:.2f}")
    print(f"Mean frame connectivity:  {results['mean_frame_connectivity']:.2f}")
    print(f"Mean matches per frame:    {results['mean_matches_per_frame']:.2f}")

    if len(inlier_percentages) > 0:
        print(f"Mean inlier percentage:    {results['mean_inlier_percentage']:.2f}%")

    print("\nLaTeX table values:")
    print(f"Number of frames & {results['number_of_frames']} \\\\")
    print(f"Number of tracks & {results['number_of_tracks']} \\\\")
    print(f"Mean track length & {results['mean_track_length']:.2f} \\\\")
    print(f"Mean frame connectivity & {results['mean_frame_connectivity']:.2f} \\\\")
    print(f"Mean matches per frame & {results['mean_matches_per_frame']:.2f} \\\\")

    if len(inlier_percentages) > 0:
        print(f"Mean inlier percentage & {results['mean_inlier_percentage']:.2f}\\% \\\\")

    return results


def pose3_camera_to_world_to_extrinsic(pose):
    """Convert a GTSAM camera-to-world Pose3 into a 3x4 world-to-camera matrix.

    Bundle poses appear to represent camera poses in frame 0 because their
    translations are used directly as camera positions. Projection requires
    world-to-camera extrinsics, so the pose is inverted here.
    """
    camera_to_world = np.asarray(pose.matrix(), dtype=float)
    world_to_camera = np.linalg.inv(camera_to_world)

    return world_to_camera[:3, :]


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


def plot_median_projection_errors(
    distances,
    median_left_errors,
    median_right_errors,
    estimator_name,
    output_filename,
    output_dir=OUTPUT_DIR,
    max_distance=50,
):
    """Plot separate median left/right errors for one estimator."""
    print("\n" + "=" * 60)
    print(f"{estimator_name} median projection error")
    print("=" * 60)

    if len(distances) == 0:
        print(f"WARNING: No {estimator_name} projection errors were computed.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.plot(
        distances,
        median_left_errors,
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Left image",
    )

    plt.plot(
        distances,
        median_right_errors,
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Right image",
    )

    plt.xlim(0, max_distance)
    plt.xticks(np.arange(0, max_distance + 1, 5))

    plt.xlabel("Distance from triangulation frame [frames]")
    plt.ylabel("Median projection error [pixels]")
    plt.title(f"{estimator_name} Median Projection Error vs. Distance from Triangulation Frame")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / output_filename

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"{estimator_name} projection-error plot saved to {output_path}")

    return output_path


def run_pnp_analysis(database):
    """
    Compute temporal PnP reprojection errors using only estimated camera poses.

    A point is triangulated in the latest camera of each track and propagated
    into earlier cameras through the composed estimated camera motions.
    """
    print("\n" + "=" * 60)
    print("PnP temporal projection-error analysis")
    print("=" * 60)

    pnp_extrinsics = np.asarray(
        np.load(GLOBAL_CAMERA_MATRICES_PATH),
        dtype=float,
    )

    estimated_world_to_camera_by_frame = {
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
        world_to_camera_by_frame=(estimated_world_to_camera_by_frame),
        calibration=calibration,
        optimize_landmarks=False,
        min_track_length=2,
        max_distance=50,
    )

    plot_path = plot_median_projection_errors(
        distances=distances,
        median_left_errors=median_left_errors,
        median_right_errors=median_right_errors,
        estimator_name="PnP temporal propagation",
        output_filename="pnp_temporal_projection_error.png",
        max_distance=50,
    )

    return {
        "distances": distances,
        "median_left_errors": median_left_errors,
        "median_right_errors": median_right_errors,
        "sample_counts": sample_counts,
        "plot_path": plot_path,
    }


def get_relative_camera_transform(
    world_to_camera_by_frame,
    source_frame,
    target_frame,
):
    """
    Return the estimated transform from source-camera coordinates to
    target-camera coordinates.

    Given:
        T_source_world: world -> source camera
        T_target_world: world -> target camera

    Then:
        T_target_source =
            T_target_world @ inverse(T_source_world)
    """
    source_world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[source_frame])
    target_world_to_camera = to_homogeneous_transform(world_to_camera_by_frame[target_frame])

    return target_world_to_camera @ np.linalg.inv(source_world_to_camera)


def compose_camera_transform(
    world_to_camera_by_frame,
    source_frame,
    target_frame,
):
    """
    Compose all consecutive estimated camera transforms from source_frame
    to target_frame.

    The returned transform maps a point from source-camera coordinates into
    target-camera coordinates.
    """
    if source_frame == target_frame:
        return np.eye(4, dtype=float)

    step = 1 if target_frame > source_frame else -1

    current_frame = source_frame
    target_from_source = np.eye(4, dtype=float)

    while current_frame != target_frame:
        next_frame = current_frame + step

        if (
            current_frame not in world_to_camera_by_frame
            or next_frame not in world_to_camera_by_frame
        ):
            raise KeyError(
                f"Missing estimated camera pose for transition {current_frame} -> {next_frame}"
            )

        next_from_current = get_relative_camera_transform(
            world_to_camera_by_frame=world_to_camera_by_frame,
            source_frame=current_frame,
            target_frame=next_frame,
        )

        target_from_source = next_from_current @ target_from_source

        current_frame = next_frame

    return target_from_source


def run_bundle_adjustment_analysis(database):
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

    plot_path = plot_median_projection_errors(
        distances=distances,
        median_left_errors=median_left_errors,
        median_right_errors=median_right_errors,
        estimator_name="Bundle Adjustment",
        output_filename="bundle_projection_error.png",
        max_distance=50,
    )

    return {
        "distances": distances,
        "median_left_errors": median_left_errors,
        "median_right_errors": median_right_errors,
        "sample_counts": sample_counts,
        "plot_path": plot_path,
    }


def run_pose_graph_analysis(database):
    pass


def run_loop_closure_analysis(database):
    pass


# def save_summary_statistics(results):
#     """Save available analysis results as JSON."""
#     SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
#
#     with open(SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
#         json.dump(results, summary_file, indent=4)
#
#     print(f"\nSummary statistics saved to {SUMMARY_PATH}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    database = build_or_load_database()
    inlier_percentages = load_inlier_percentages()

    tracking_results = run_tracking_analysis(
        database,
        inlier_percentages,
    )

    # pnp_results = run_pnp_analysis(database)

    # Keyframes and local BA are computed only once.
    ba_context = prepare_bundle_adjustment_analysis(
        database
    )

    bundle_window_error_path = (
        plot_bundle_window_average_errors(
            ba_context.solutions
        )
    )

    print(
        f"Bundle-window average-error plot: "
        f"{bundle_window_error_path}"
    )

    print("\n" + "=" * 60)
    print(f"All analysis outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
