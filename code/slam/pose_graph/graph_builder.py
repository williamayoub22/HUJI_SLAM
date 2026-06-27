from dataclasses import dataclass

import gtsam
from gtsam.symbol_shorthand import C
from slam.pose_graph.covariance_routing import (
    CovarianceGraph,
    symmetrize_covariance,
)

from slam.pose_graph.constraints import RelativePoseConstraint


@dataclass
class PoseGraphBuildResult:
    graph: gtsam.NonlinearFactorGraph
    initial_estimates: gtsam.Values
    keyframe_ids: list[int]
    covariance_graph: CovarianceGraph


def build_pose_graph(
    constraints: list[RelativePoseConstraint],
    first_pose: gtsam.Pose3,
    prior_sigma: float = 1e-6,
) -> PoseGraphBuildResult:
    """Build the GTSAM pose graph and its covariance-routing counterpart."""
    if not constraints:
        raise ValueError("Cannot build a pose graph without relative constraints.")

    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    covariance_graph = CovarianceGraph()

    constraints = sorted(
        constraints,
        key=lambda constraint: constraint.start_frame,
    )

    first_frame = constraints[0].start_frame
    initial_estimates.insert(C(first_frame), first_pose)

    prior_noise = gtsam.noiseModel.Isotropic.Sigma(
        6,
        prior_sigma,
    )
    graph.add(
        gtsam.PriorFactorPose3(
            C(first_frame),
            first_pose,
            prior_noise,
        )
    )

    keyframe_ids = [first_frame]

    for constraint in constraints:
        start_key = C(constraint.start_frame)
        end_key = C(constraint.end_frame)

        if not initial_estimates.exists(start_key):
            raise ValueError(
                "Pose-graph constraints do not form a consecutive chain. "
                f"Missing initial estimate for frame {constraint.start_frame}."
            )

        covariance = symmetrize_covariance(constraint.covariance)

        noise_model = gtsam.noiseModel.Gaussian.Covariance(
            covariance,
        )

        graph.add(
            gtsam.BetweenFactorPose3(
                start_key,
                end_key,
                constraint.relative_pose,
                noise_model,
            )
        )

        covariance_graph.add_edge(
            start_frame=constraint.start_frame,
            end_frame=constraint.end_frame,
            covariance=covariance,
        )

        if not initial_estimates.exists(end_key):
            start_pose = initial_estimates.atPose3(start_key)
            end_pose = start_pose.compose(
                constraint.relative_pose,
            )
            initial_estimates.insert(end_key, end_pose)
            keyframe_ids.append(constraint.end_frame)

    return PoseGraphBuildResult(
        graph=graph,
        initial_estimates=initial_estimates,
        keyframe_ids=keyframe_ids,
        covariance_graph=covariance_graph,
    )
