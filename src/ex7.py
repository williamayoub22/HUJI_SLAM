# Full ex7 pipeline:
# The full pipeline will be:
#
# 1. Build the original pose graph from Exercise 6.
# 2. For each current keyframe c_n, consider older keyframes c_i.
# 3. Estimate how uncertain the relative pose c_i \rightarrow c_n is using the graph.
# 4. Keep plausible candidates.
# 5. Try visual matching and PnP/RANSAC for those candidates.
# 6. Add a BetweenFactorPose3 for accepted loops.
# 7. Re-optimize the pose graph.
# 8. Show that the trajectory and its uncertainty improve.

from slam.ba.gtsam_utils import pose3_from_world_to_camera_extrinsic
from slam.pipeline.pose_graph_pipeline import (
    load_ex6_inputs,
    solve_bundle_windows_and_extract_constraints,
)
from slam.pose_graph.covariance_routing import (
    detect_candidates_for_keyframe,
)
from slam.pose_graph.graph_builder import build_pose_graph
from slam.pose_graph.optimization import optimize_pose_graph


def q_1():
    """
    Section 7.1: detect possible loop-closure candidates.

    For every keyframe c_n, test all sufficiently earlier keyframes c_i using

        t2v(C_i^{-1} C_n)^T Sigma_{n|i}^{-1} t2v(C_i^{-1} C_n).

    Returns:
        candidates_by_target:
            Maps each target frame c_n to its accepted earlier candidates.
    """
    chi_square_threshold = 12.592
    # min_keyframe_separation = 5
    #
    # db, world_to_camera_extrinsics, calibration = load_ex6_inputs()
    #
    # _, constraints = solve_bundle_windows_and_extract_constraints(
    #     db=db,
    #     world_to_camera_extrinsics=world_to_camera_extrinsics,
    #     calibration=calibration,
    #     keyframe_step=10,
    #     verbose=False,
    # )
    #
    # if not constraints:
    #     raise RuntimeError("No relative-pose constraints were extracted.")
    #
    # first_frame = constraints[0].start_frame
    # first_pose = pose3_from_world_to_camera_extrinsic(
    #     world_to_camera_extrinsics[first_frame]
    # )
    #
    # build_result = build_pose_graph(
    #     constraints=constraints,
    #     first_pose=first_pose,
    # )
    #
    # pose_graph_result = optimize_pose_graph(
    #     graph=build_result.graph,
    #     initial_estimates=build_result.initial_estimates,
    #     keyframe_ids=build_result.keyframe_ids,
    # )
    #
    # keyframe_ids = build_result.keyframe_ids
    #
    # candidates_by_target = {}
    # all_candidates = []
    #
    # for target_frame in keyframe_ids:
    #     candidates = detect_candidates_for_keyframe(
    #         optimized_values=pose_graph_result.optimized_estimates,
    #         covariance_graph=build_result.covariance_graph,
    #         keyframe_ids=keyframe_ids,
    #         target_frame=target_frame,
    #         min_keyframe_separation=min_keyframe_separation,
    #         chi_square_threshold=chi_square_threshold,
    #     )
    #
    #     if candidates:
    #         candidates_by_target[target_frame] = candidates
    #         all_candidates.extend(candidates)
    #
    # print("\n" + "=" * 60)
    # print("[7.1] Mahalanobis Loop-Closure Candidate Detection")
    # print("=" * 60)
    # print(f"Keyframes tested:              {len(keyframe_ids)}")
    # print(f"Minimum temporal separation:   {min_keyframe_separation}")
    # print(f"Mahalanobis threshold:         {chi_square_threshold:.3f}")
    # print(f"Keyframes with candidates:     {len(candidates_by_target)}")
    # print(f"Accepted candidate pairs:      {len(all_candidates)}")
    #
    # if len(keyframe_ids) != 310:
    #     print(
    #         "Warning: expected 310 keyframes according to the exercise, "
    #         f"but built a graph with {len(keyframe_ids)}."
    #     )
    #
    # if all_candidates:
    #     best_candidates = sorted(
    #         all_candidates,
    #         key=lambda candidate: candidate.mahalanobis_squared,
    #     )[:10]
    #
    #     print("\nBest accepted candidates:")
    #     for candidate in best_candidates:
    #         print(
    #             f"  c_{candidate.target_frame} <-> c_{candidate.source_frame}: "
    #             f"d^2={candidate.mahalanobis_squared:.3f}, "
    #             f"path={candidate.path_frame_ids}"
    #         )
    # else:
    #     print("\nNo candidate pairs passed the Mahalanobis threshold.")
    #
    # return candidates_by_target

def main() -> None:
    # print("1")
    q_1()


if __name__ == "__main__":
    main()
