"""
Provides pose graph diagnostics components and utilities for the SLAM pipeline.
"""

import numpy as np


def format_relative_constraint_report(
    constraint,
) -> str:
    covariance_eigenvalues = np.linalg.eigvalsh(
        constraint.covariance,
    )

    return "\n".join(
        [
            "=" * 60,
            "[6.1] Relative pose constraint from first BA window",
            "=" * 60,
            f"Keyframes: {constraint.start_frame} -> {constraint.end_frame}",
            "",
            "Relative pose:",
            str(constraint.relative_pose),
            "",
            "Conditional relative covariance Cov(c_end | c_start):",
            str(constraint.covariance),
            "",
            "Translation covariance block:",
            str(constraint.covariance[3:6, 3:6]),
            "",
            "Covariance eigenvalues:",
            str(covariance_eigenvalues),
        ]
    )


def format_pose_graph_report(
    initial_error: float,
    final_error: float,
    num_factors: int,
    num_poses: int,
) -> str:
    return "\n".join(
        [
            "=" * 60,
            "[6.2] Global Pose Graph Optimization",
            "=" * 60,
            f"Factors: {num_factors}",
            f"Poses: {num_poses}",
            f"Initial error: {initial_error:.8f}",
            f"Final error:   {final_error:.8f}",
        ]
    )
