from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass
from gtsam.symbol_shorthand import C
import gtsam
import numpy as np

from slam.pose_graph.constraints import symmetrize


def covariance_volume(covariance: np.ndarray) -> float:
    """
    Return the uncertainty-volume score det(Sigma).

    For a Gaussian covariance ellipsoid, this is proportional to its volume sqrt(det(Sigma)).
    """
    covariance = symmetrize(covariance)

    determinant = float(np.linalg.det(covariance))

    if determinant < -1e-10:
        raise ValueError(
            "Covariance determinant is negative. "
            f"Expected a positive-semidefinite covariance, got {determinant}."
        )

    return max(determinant, 0.0)


def sum_covariances(covariances: list[np.ndarray]) -> np.ndarray:
    """Approximate path covariance by summing its edge covariances."""
    total = np.zeros((6, 6), dtype=float)

    for covariance in covariances:
        total += symmetrize(covariance)

    return symmetrize(total)


@dataclass(frozen=True)
class CovarianceEdge:
    """Directed relative-pose measurement edge."""

    neighbor_frame: int
    covariance: np.ndarray


@dataclass(frozen=True)
class CovariancePath:
    """Minimum-volume covariance path."""

    frame_ids: list[int]
    covariance: np.ndarray
    cost: float


class CovarianceGraph:
    """
    Directed graph for the covariance-valued shortest-path approximation.

    Each edge u -> v stores the 6x6 covariance of the relative measurement.
    Matrix addition accumulates path covariances. Paths are ranked by det(accumulated covariance).
    """

    def __init__(self) -> None:
        self._adjacency: dict[int, list[CovarianceEdge]] = defaultdict(list)

    def add_directed_edge(
        self,
        source_frame: int,
        target_frame: int,
        covariance: np.ndarray,
    ) -> None:
        """Add the relative-pose covariance for source_frame -> target_frame."""
        covariance = symmetrize(covariance)

        self._adjacency[source_frame].append(
            CovarianceEdge(
                neighbor_frame=target_frame,
                covariance=covariance,
            )
        )

    def neighbors(self, frame_id: int) -> list[CovarianceEdge]:
        """Return all outgoing covariance edges from one keyframe."""
        return self._adjacency.get(frame_id, [])

    def shortest_path(
        self,
        source_frame: int,
        target_frame: int,
    ) -> CovariancePath | None:
        """
        Return the path with minimum accumulated covariance volume.

        For a candidate extension u -> v:
            Sigma_candidate = Sigma[u] + Sigma_uv

        and its priority is: det(Sigma_candidate).
        """
        if source_frame == target_frame:
            return CovariancePath(
                frame_ids=[source_frame],
                covariance=np.zeros((6, 6), dtype=float),
                cost=0.0,
            )

        if source_frame not in self._adjacency:
            return None

        zero_covariance = np.zeros((6, 6), dtype=float)

        best_covariances: dict[int, np.ndarray] = {
            source_frame: zero_covariance,
        }
        best_costs: dict[int, float] = {
            source_frame: 0.0,
        }

        parents: dict[int, tuple[int, np.ndarray]] = {}

        insertion_counter = itertools.count()
        priority_queue: list[tuple[float, int, int]] = [
            (0.0, next(insertion_counter), source_frame),
        ]

        while priority_queue:
            current_cost, _, current_frame = heapq.heappop(priority_queue)

            if current_cost > best_costs[current_frame]:
                continue

            if current_frame == target_frame:
                break

            current_covariance = best_covariances[current_frame]

            for edge in self.neighbors(current_frame):
                candidate_covariance = symmetrize(current_covariance + edge.covariance)
                candidate_cost = covariance_volume(candidate_covariance)

                previous_cost = best_costs.get(
                    edge.neighbor_frame,
                    float("inf"),
                )

                if candidate_cost < previous_cost:
                    best_covariances[edge.neighbor_frame] = candidate_covariance
                    best_costs[edge.neighbor_frame] = candidate_cost
                    parents[edge.neighbor_frame] = (
                        current_frame,
                        edge.covariance,
                    )

                    heapq.heappush(
                        priority_queue,
                        (
                            candidate_cost,
                            next(insertion_counter),
                            edge.neighbor_frame,
                        ),
                    )

        if target_frame not in best_covariances:
            return None

        frame_ids = [target_frame]
        current_frame = target_frame

        while current_frame != source_frame:
            parent_frame, _ = parents[current_frame]
            frame_ids.append(parent_frame)
            current_frame = parent_frame

        frame_ids.reverse()

        return CovariancePath(
            frame_ids=frame_ids,
            covariance=best_covariances[target_frame],
            cost=best_costs[target_frame],
        )

@dataclass(frozen=True)
class LoopClosureCandidate:
    """A candidate loop closure between an earlier keyframe and a later one."""

    source_frame: int
    target_frame: int
    relative_pose: gtsam.Pose3
    relative_delta: np.ndarray
    covariance: np.ndarray
    mahalanobis_squared: float
    path_frame_ids: list[int]


def mahalanobis_squared(
    delta: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Compute delta^T Sigma^{-1} delta without explicitly inverting Sigma.
    """
    delta = np.asarray(delta, dtype=float).reshape(-1)
    covariance = symmetrize(covariance)

    if delta.shape != (6,):
        raise ValueError(
            f"Expected a 6D pose perturbation, got shape {delta.shape}."
        )

    try:
        value = float(delta @ np.linalg.solve(covariance, delta))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Cannot compute Mahalanobis distance: covariance is singular."
        ) from error

    # Small negative values can appear from numerical error.
    if value < -1e-10:
        raise ValueError(
            f"Mahalanobis distance is negative ({value}); "
            "check the covariance matrix."
        )

    return max(value, 0.0)

def t2v(relative_pose: gtsam.Pose3) -> np.ndarray:
    """
    Map a relative Pose3 transform to GTSAM's local 6D pose coordinates.

    This is the implementation counterpart of the course notation
    t2v(C_i^{-1} C_n). It must use the same local coordinate convention as
    the covariance matrices returned by GTSAM.
    """
    return np.asarray(
        gtsam.Pose3.Identity().localCoordinates(relative_pose),
        dtype=float,
    ).reshape(6)


def detect_candidates_for_keyframe(
    *,
    optimized_values: gtsam.Values,
    covariance_graph: CovarianceGraph,
    keyframe_ids: list[int],
    target_frame: int,
    min_keyframe_separation: int = 5,
    chi_square_threshold: float = 12.592,
    max_candidates: int | None = None,
) -> list[LoopClosureCandidate]:
    """
    Find earlier keyframes that are plausible loop-closure candidates for c_n.

    For every eligible earlier keyframe c_i:
      1. Find the minimum-uncertainty path i -> n.
      2. Use its accumulated covariance as Sigma_{n|i}.
      3. Compute Delta c_ni = t2v(C_i^{-1} C_n).
      4. Retain the pair if

             Delta c_ni^T Sigma_{n|i}^{-1} Delta c_ni

         is below the chi-square threshold.

    Temporal separation is measured in positions in keyframe_ids rather than
    raw frame indices, so it remains meaningful if the keyframe interval
    changes.
    """
    if len(keyframe_ids) != len(set(keyframe_ids)):
        raise ValueError("keyframe_ids must not contain duplicates.")

    if target_frame not in keyframe_ids:
        raise ValueError(
            f"Target frame {target_frame} is not a pose-graph keyframe."
        )

    if min_keyframe_separation < 1:
        raise ValueError(
            "min_keyframe_separation must be at least 1."
        )

    if chi_square_threshold <= 0:
        raise ValueError(
            "chi_square_threshold must be positive."
        )

    if max_candidates is not None and max_candidates < 1:
        raise ValueError(
            "max_candidates must be positive or None."
        )

    target_index = keyframe_ids.index(target_frame)

    # Nearby temporal frames are expected to be close and are not loop closures.
    last_candidate_index = target_index - min_keyframe_separation

    if last_candidate_index < 0:
        return []

    target_key = C(target_frame)

    if not optimized_values.exists(target_key):
        raise ValueError(
            f"Missing optimized pose for target frame {target_frame}."
        )

    target_pose = optimized_values.atPose3(target_key)
    candidates: list[LoopClosureCandidate] = []

    for source_frame in keyframe_ids[: last_candidate_index + 1]:
        source_key = C(source_frame)

        if not optimized_values.exists(source_key):
            raise ValueError(
                f"Missing optimized pose for source frame {source_frame}."
            )

        covariance_path = covariance_graph.shortest_path(
            source_frame=source_frame,
            target_frame=target_frame,
        )

        if covariance_path is None:
            continue

        source_pose = optimized_values.atPose3(source_key)

        # C_i^{-1} C_n.
        relative_pose = source_pose.between(target_pose)

        # Delta c_ni = t2v(C_i^{-1} C_n).
        relative_delta = t2v(relative_pose)

        distance_squared = mahalanobis_squared(
            delta=relative_delta,
            covariance=covariance_path.covariance,
        )

        if distance_squared > chi_square_threshold:
            continue

        candidates.append(
            LoopClosureCandidate(
                source_frame=source_frame,
                target_frame=target_frame,
                relative_pose=relative_pose,
                relative_delta=relative_delta,
                covariance=covariance_path.covariance,
                mahalanobis_squared=distance_squared,
                path_frame_ids=covariance_path.frame_ids,
            )
        )

    candidates.sort(
        key=lambda candidate: candidate.mahalanobis_squared
    )

    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    return candidates
