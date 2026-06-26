# weighted graph + Dijkstra's algorithm for relative-covariance estimation

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq

import gtsam
import numpy as np


def symmetrize_covariance(covariance: np.ndarray) -> np.ndarray:
    """Return a numerically symmetric 6x6 covariance matrix."""
    covariance = np.asarray(covariance, dtype=float)

    if covariance.shape != (6, 6):
        raise ValueError(
            f"Expected covariance shape (6, 6), got {covariance.shape}."
        )

    return 0.5 * (covariance + covariance.T)


def covariance_trace(covariance: np.ndarray) -> float:
    """
    Return the scalar uncertainty cost of one edge.

    The trace is additive under the Exercise 7 approximation:
    trace(sum Sigma_e) = sum trace(Sigma_e).
    """
    covariance = symmetrize_covariance(covariance)
    cost = float(np.trace(covariance))

    # A valid covariance should be positive semidefinite, so its trace should
    # be nonnegative. Small negative values can arise from numerical noise.
    if cost < -1e-10:
        raise ValueError(
            f"Covariance trace must be nonnegative, got {cost}."
        )

    return max(cost, 0.0)


def sum_covariances(covariances: list[np.ndarray]) -> np.ndarray:
    """Approximate relative covariance by summing path-edge covariances."""
    total = np.zeros((6, 6), dtype=float)

    for covariance in covariances:
        total += symmetrize_covariance(covariance)

    return symmetrize_covariance(total)


@dataclass(frozen=True)
class CovarianceEdge:
    neighbor_frame: int
    relative_pose: gtsam.Pose3
    covariance: np.ndarray


@dataclass(frozen=True)
class CovariancePath:
    """Minimum-uncertainty path and its accumulated covariance."""

    frame_ids: list[int]
    covariance: np.ndarray
    cost: float


class CovarianceGraph:
    """
    Lightweight undirected graph for uncertainty-aware path search.

    This is intentionally separate from GTSAM's factor graph. It contains
    only relative-pose edges between keyframes and their covariances.
    """

    def __init__(self) -> None:
        self._adjacency: dict[int, list[CovarianceEdge]] = defaultdict(list)

    def add_edge(
        self,
        start_frame: int,
        end_frame: int,
        covariance: np.ndarray,
    ) -> None:
        covariance = symmetrize_covariance(covariance)

        self._adjacency[start_frame].append(
            CovarianceEdge(
                neighbor_frame=end_frame,
                covariance=covariance,
            )
        )
        self._adjacency[end_frame].append(
            CovarianceEdge(
                neighbor_frame=start_frame,
                covariance=covariance,
            )
        )

    def neighbors(self, frame_id: int) -> list[CovarianceEdge]:
        """Return all relative-pose edges incident to one keyframe."""
        return self._adjacency.get(frame_id, [])

    def shortest_path(
        self,
        source_frame: int,
        target_frame: int,
    ) -> CovariancePath | None:
        """
        Find the minimum-uncertainty path using Dijkstra's algorithm.

        Dijkstra minimizes the sum of trace(covariance) edge costs. After
        finding the path, the method sums the original 6x6 covariances along
        that path to obtain the approximation of relative covariance.
        """
        if source_frame == target_frame:
            return CovariancePath(
                frame_ids=[source_frame],
                covariance=np.zeros((6, 6), dtype=float),
                cost=0.0,
            )

        if source_frame not in self._adjacency:
            return None

        # Best scalar uncertainty cost found so far for each frame.
        distances: dict[int, float] = {
            source_frame: 0.0,
        }

        # child_frame -> (parent_frame, covariance of parent -> child edge)
        parents: dict[int, tuple[int, np.ndarray]] = {}

        # Entries are (current_cost, frame_id).
        priority_queue: list[tuple[float, int]] = [
            (0.0, source_frame),
        ]

        while priority_queue:
            current_cost, current_frame = heapq.heappop(priority_queue)

            # Ignore stale queue entries superseded by a better path.
            if current_cost > distances[current_frame]:
                continue

            if current_frame == target_frame:
                break

            for edge in self.neighbors(current_frame):
                edge_cost = covariance_trace(edge.covariance)
                candidate_cost = current_cost + edge_cost

                if candidate_cost < distances.get(
                    edge.neighbor_frame,
                    float("inf"),
                ):
                    distances[edge.neighbor_frame] = candidate_cost
                    parents[edge.neighbor_frame] = (
                        current_frame,
                        edge.covariance,
                    )

                    heapq.heappush(
                        priority_queue,
                        (candidate_cost, edge.neighbor_frame),
                    )

        if target_frame not in distances:
            return None

        path_frame_ids = [target_frame]
        path_covariances: list[np.ndarray] = []

        current_frame = target_frame

        while current_frame != source_frame:
            parent_frame, edge_covariance = parents[current_frame]

            path_frame_ids.append(parent_frame)
            path_covariances.append(edge_covariance)

            current_frame = parent_frame

        path_frame_ids.reverse()

        return CovariancePath(
            frame_ids=path_frame_ids,
            covariance=sum_covariances(path_covariances),
            cost=distances[target_frame],
        )
