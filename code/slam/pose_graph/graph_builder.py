import gtsam
from gtsam.symbol_shorthand import C

from slam.pose_graph.constraints import RelativePoseConstraint


def build_pose_graph(
    constraints: list[RelativePoseConstraint],
    first_pose: gtsam.Pose3,
    prior_sigma: float = 1e-6,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, list[int]]:
    """Builds a pose graph containing:
    - a prior on the first keyframe,
    - one BetweenFactorPose3 per BA-derived relative constraint,
    - an initial estimate obtained by chaining relative motions.
    """
    if not constraints:
        raise ValueError("Cannot build a pose graph without relative constraints.")

    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

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

        covariance = 0.5 * (constraint.covariance + constraint.covariance.T)

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

        if not initial_estimates.exists(end_key):
            start_pose = initial_estimates.atPose3(start_key)
            end_pose = start_pose.compose(
                constraint.relative_pose,
            )
            initial_estimates.insert(end_key, end_pose)
            keyframe_ids.append(constraint.end_frame)

    return graph, initial_estimates, keyframe_ids
