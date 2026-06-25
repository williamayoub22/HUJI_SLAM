from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import C

from slam.ba.results import BundleWindowSolution


@dataclass(frozen=True)
class RelativePoseConstraint:
    """One relative keyframe measurement extracted from a BA window."""

    start_frame: int
    end_frame: int
    relative_pose: gtsam.Pose3
    covariance: np.ndarray


def conditional_covariance(
    joint_covariance: np.ndarray,
) -> np.ndarray:
    """
    Computes Cov(c_end | c_start) from the joint covariance of
    [c_start, c_end].

    The input is a 12x12 matrix:
        [ Sigma_ss  Sigma_se ]
        [ Sigma_es  Sigma_ee ]
    """
    if joint_covariance.shape != (12, 12):
        raise ValueError(
            "Expected a 12x12 joint covariance for two Pose3 variables, "
            f"got {joint_covariance.shape}."
        )

    sigma_start_start = joint_covariance[:6, :6]
    sigma_start_end = joint_covariance[:6, 6:]
    sigma_end_start = joint_covariance[6:, :6]
    sigma_end_end = joint_covariance[6:, 6:]

    covariance = sigma_end_end - sigma_end_start @ np.linalg.solve(
        sigma_start_start, sigma_start_end
    )

    # Numerical cleanup: a covariance must be symmetric.
    return 0.5 * (covariance + covariance.T)


def extract_relative_pose_constraint(
    solution: BundleWindowSolution,
) -> RelativePoseConstraint:
    """
    Extracts the relative pose and conditional covariance between the
    first and last keyframes of one optimized BA window.
    """
    result = solution.result

    start_key = C(solution.start_frame)
    end_key = C(solution.end_frame)

    start_pose = result.optimized.atPose3(start_key)
    end_pose = result.optimized.atPose3(end_key)

    relative_pose = start_pose.between(end_pose)

    marginals = gtsam.Marginals(
        result.graph,
        result.optimized,
    )

    keys = gtsam.KeyVector()
    keys.append(start_key)
    keys.append(end_key)

    joint_covariance = marginals.jointMarginalCovariance(keys).fullMatrix()

    covariance = conditional_covariance(joint_covariance)

    return RelativePoseConstraint(
        start_frame=solution.start_frame,
        end_frame=solution.end_frame,
        relative_pose=relative_pose,
        covariance=covariance,
    )


import numpy as np
from gtsam.symbol_shorthand import C


def verify_relative_measurement(
    bundle_solution,
    constraint: RelativePoseConstraint,
    tolerance: float = 1e-6,
) -> None:
    """
    Verifies that the extracted relative pose reconstructs the optimized
    final keyframe pose from the optimized initial keyframe pose.
    """
    optimized = bundle_solution.result.optimized

    start_pose = optimized.atPose3(
        C(bundle_solution.start_frame),
    )
    end_pose = optimized.atPose3(
        C(bundle_solution.end_frame),
    )

    reconstructed_end_pose = start_pose.compose(
        constraint.relative_pose,
    )

    if not reconstructed_end_pose.equals(
        end_pose,
        tolerance,
    ):
        raise AssertionError(
            "Relative pose does not reconstruct the optimized end pose. "
            "Check pose direction or compose order."
        )

    if not np.allclose(
        constraint.covariance,
        constraint.covariance.T,
        atol=1e-8,
    ):
        raise AssertionError("Relative covariance is not symmetric.")
