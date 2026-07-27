"""
Provides constraints components and utilities for the SLAM pipeline.
"""

from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import C

from src.slam.bundle_adjustment.types import BundleWindowSolution


@dataclass(frozen=True)
class RelativePoseConstraint:
    """One relative keyframe measurement extracted from a BA window."""

    start_frame: int
    end_frame: int
    relative_pose: gtsam.Pose3
    covariance: np.ndarray
    information: np.ndarray


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Remove small numerical asymmetries."""
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def conditional_information_from_joint_information(
    joint_information: np.ndarray,
) -> np.ndarray:
    """Extract Ω_end|start from the joint canonical representation."""
    joint_information = np.asarray(joint_information, dtype=float)

    if joint_information.shape != (12, 12):
        raise ValueError(
            "Expected a 12x12 joint information matrix for two Pose3 "
            f"variables, got {joint_information.shape}."
        )

    joint_information = symmetrize(joint_information)  # todo: needed?
    information = joint_information[:6, :6]
    return symmetrize(information)


def covariance_from_information(
    information: np.ndarray,
) -> np.ndarray:
    """Compute Σ = Ω^{-1} for one 6D relative-pose factor."""
    information = np.asarray(information, dtype=float)

    if information.shape != (6, 6):
        raise ValueError(f"Expected a 6x6 information matrix, got {information.shape}.")

    covariance = np.linalg.inv(information)
    return symmetrize(covariance)


def extract_relative_pose_constraint(
    solution: BundleWindowSolution,
) -> RelativePoseConstraint:
    """Extract the relative-pose factor between the first and last keyframes
    of one optimized BA window.

    The factor models p(c_end | c_start), so we query the joint information
    in [end, start] order and extract Λ_end,end.
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

    keys.append(end_key)
    keys.append(start_key)

    joint_information = marginals.jointMarginalInformation(keys).fullMatrix()

    information = conditional_information_from_joint_information(joint_information)
    covariance = covariance_from_information(information)

    return RelativePoseConstraint(
        start_frame=solution.start_frame,
        end_frame=solution.end_frame,
        relative_pose=relative_pose,
        covariance=covariance,
        information=information,
    )
