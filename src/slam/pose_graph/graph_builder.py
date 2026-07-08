from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import C

from src.slam.pose_graph.constraints import RelativePoseConstraint, symmetrize
from src.slam.pose_graph.covariance_routing import (
    CovarianceGraph,
)


@dataclass
class PoseGraphBuildResult:
    """Pose graph together with data required for loop-closure detection."""

    graph: gtsam.NonlinearFactorGraph
    initial_estimates: gtsam.Values
    keyframe_ids: list[int]
    covariance_graph: CovarianceGraph


def build_pose_graph(
    constraints: list[RelativePoseConstraint],
    first_pose: gtsam.Pose3,
    prior_sigma: float = 1e-6,
) -> PoseGraphBuildResult:
    """Build the pose graph from consecutive relative-pose constraints.

    Each GTSAM BetweenFactor uses the conditional information matrix extracted
    from Bundle Adjustment. The covariance-routing graph retains the matching
    covariance matrix for later path-based relative-covariance estimation.
    """
    if not constraints:
        raise ValueError("Cannot build a pose graph without relative constraints.")

    constraints = sorted(
        constraints,
        key=lambda constraint: constraint.start_frame,
    )

    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    covariance_graph = CovarianceGraph()

    first_frame = constraints[0].start_frame
    first_key = C(first_frame)

    initial_estimates.insert(first_key, first_pose)

    prior_noise = gtsam.noiseModel.Isotropic.Sigma(
        6,
        prior_sigma,
    )

    graph.add(
        gtsam.PriorFactorPose3(
            first_key,
            first_pose,
            prior_noise,
        )
    )

    keyframe_ids = [first_frame]

    for constraint in constraints:
        start_frame = constraint.start_frame
        end_frame = constraint.end_frame

        start_key = C(start_frame)
        end_key = C(end_frame)

        if not initial_estimates.exists(start_key):
            raise ValueError(
                "Pose-graph constraints must form a consecutive chain. "
                f"Missing initial estimate for frame {start_frame}."
            )

        covariance = symmetrize(
            constraint.covariance,
        )
        # [FIX] The "Titanium" Odometry
        # Inject a minimum uncertainty "floor" into the odometry constraints.
        # Use 1e-8 for rotation (tight) and 1e-4 for translation (flexible)
        # to prevent massive lever-arm uncertainty explosions.
        covariance += np.diag([1e-8, 1e-8, 1e-8, 1e-4, 1e-4, 1e-4])

        information = np.linalg.inv(covariance)
        information = 0.5 * (information + information.T)

        # The factor models p(c_end | c_start). Its information matrix is the
        # Lambda_end,end block from the joint canonical representation.
        noise_model = gtsam.noiseModel.Gaussian.Information(
            information,
        )

        graph.add(
            gtsam.BetweenFactorPose3(
                start_key,
                end_key,
                constraint.relative_pose,
                noise_model,
            )
        )

        # Keep the covariance form for the covariance-valued shortest-path
        # approximation used during loop-closure candidate detection.
        covariance_graph.add_directed_edge(
            source_frame=start_frame,
            target_frame=end_frame,
            covariance=covariance,
        )

        if not initial_estimates.exists(end_key):
            start_pose = initial_estimates.atPose3(start_key)
            end_pose = start_pose.compose(
                constraint.relative_pose,
            )

            initial_estimates.insert(
                end_key,
                end_pose,
            )
            keyframe_ids.append(end_frame)

    # validate_constraints(
    #     constraints=constraints,
    #     initial_estimates=initial_estimates,
    # )

    return PoseGraphBuildResult(
        graph=graph,
        initial_estimates=initial_estimates,
        keyframe_ids=keyframe_ids,
        covariance_graph=covariance_graph,
    )


# DEBUG:
def validate_constraints(
    constraints: list[RelativePoseConstraint],
    initial_estimates: gtsam.Values,
) -> None:
    """Verify that each relative constraint, its covariance, and its information
    matrix are mutually consistent with the initial pose chain.
    """
    print("\n" + "=" * 60)
    print("Pose-graph constraint consistency checks")
    print("=" * 60)

    max_inverse_error = 0.0
    max_pose_residual = 0.0
    min_cov_eigenvalue = float("inf")
    max_cov_condition = 0.0

    for index, constraint in enumerate(constraints):
        covariance = symmetrize(np.asarray(constraint.covariance, dtype=float))
        information = np.asarray(
            constraint.information,
            dtype=float,
        )
        information = 0.5 * (information + information.T)

        if covariance.shape != (6, 6):
            raise ValueError(
                f"Constraint {index}: covariance shape is {covariance.shape}, expected (6, 6)."
            )

        if information.shape != (6, 6):
            raise ValueError(
                f"Constraint {index}: information shape is {information.shape}, expected (6, 6)."
            )

        if not np.all(np.isfinite(covariance)):
            raise ValueError(f"Constraint {index}: covariance has NaN/inf.")

        if not np.all(np.isfinite(information)):
            raise ValueError(f"Constraint {index}: information has NaN/inf.")

        covariance_eigenvalues = np.linalg.eigvalsh(covariance)
        information_eigenvalues = np.linalg.eigvalsh(information)

        if np.min(covariance_eigenvalues) <= 0:
            raise ValueError(
                f"Constraint {index}: covariance is not positive definite; "
                f"min eigenvalue={np.min(covariance_eigenvalues):.3e}"
            )

        if np.min(information_eigenvalues) <= 0:
            raise ValueError(
                f"Constraint {index}: information is not positive definite; "
                f"min eigenvalue={np.min(information_eigenvalues):.3e}"
            )

        inverse_error = np.linalg.norm(
            covariance @ information - np.eye(6),
            ord="fro",
        )

        condition_number = np.linalg.cond(covariance)

        start_pose = initial_estimates.atPose3(C(constraint.start_frame))
        end_pose = initial_estimates.atPose3(C(constraint.end_frame))

        predicted_relative_pose = start_pose.between(end_pose)

        pose_residual = gtsam.Pose3.Logmap(
            constraint.relative_pose.inverse().compose(predicted_relative_pose)
        )
        pose_residual_norm = float(np.linalg.norm(pose_residual))

        max_inverse_error = max(max_inverse_error, inverse_error)
        max_pose_residual = max(max_pose_residual, pose_residual_norm)
        min_cov_eigenvalue = min(
            min_cov_eigenvalue,
            float(np.min(covariance_eigenvalues)),
        )
        max_cov_condition = max(max_cov_condition, condition_number)

        if index < 5:
            print(f"\nConstraint {index}: c_{constraint.start_frame} -> c_{constraint.end_frame}")
            print(f"  covariance std: {np.sqrt(np.diag(covariance))}")
            print(f"  covariance eig: {covariance_eigenvalues}")
            print(f"  covariance cond: {condition_number:.3e}")
            print(f"  ||Sigma Lambda - I||_F: {inverse_error:.3e}")
            print(f"  initial relative-pose residual: {pose_residual_norm:.3e}")

    print("\nSummary:")
    print(f"  max ||Sigma Lambda - I||_F: {max_inverse_error:.3e}")
    print(f"  max initial pose residual:  {max_pose_residual:.3e}")
    print(f"  minimum covariance eigenvalue: {min_cov_eigenvalue:.3e}")
    print(f"  maximum covariance condition:   {max_cov_condition:.3e}")
