from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def symmetrize_covariance(covariance: np.ndarray) -> np.ndarray:
    """Return a numerically symmetric 6x6 covariance matrix."""
    covariance = np.asarray(covariance, dtype=float)

    if covariance.shape != (6, 6):
        raise ValueError(f"Expected covariance shape (6, 6), got {covariance.shape}.")

    return 0.5 * (covariance + covariance.T)


def covariance_determinant(covariance: np.ndarray) -> float:
    """Return det(covariance), the uncertainty-volume score.

    The determinant is proportional to the squared volume of the covariance
    ellipsoid. Smaller determinant means a more certain accumulated path.
    """
    covariance = symmetrize_covariance(covariance)

    determinant = float(np.linalg.det(covariance))

    # Small negative values may occur because of floating-point error.
    if determinant < -1e-10:
        raise ValueError(
            "Covariance determinant is negative. "
            f"Expected a positive-semidefinite covariance, got {determinant}."
        )

    return max(determinant, 0.0)


def sum_covariances(covariances: list[np.ndarray]) -> np.ndarray:
    """Approximate relative covariance by summing the directed edge covariances
    along a selected path.
    """
    total = np.zeros((6, 6), dtype=float)

    for covariance in covariances:
        total += symmetrize_covariance(covariance)

    return symmetrize_covariance(total)


@dataclass(frozen=True)
class CovarianceEdge:
    """A directed relative-pose measurement edge.

    covariance represents the uncertainty of the measurement:
        source_frame -> neighbor_frame
    """
    neighbor_frame: int
    covariance: np.ndarray


@dataclass(frozen=True)
class CovariancePath:
    """Minimum-uncertainty directed path and its accumulated covariance."""
    frame_ids: list[int]
    covariance: np.ndarray
    cost: float


class CovarianceGraph:
    """Directed graph for covariance-aware shortest-path search.

    Each edge u -> v stores w(u, v), namely the covariance of the relative
    measurement from u to v. Dijkstra stores an accumulated covariance
    Sigma[v] for each vertex and compares paths using det(Sigma[v]).
    """

    def __init__(self) -> None:
        self._adjacency: dict[int, list[CovarianceEdge]] = defaultdict(list)

    def add_directed_edge(
        self,
        source_frame: int,
        target_frame: int,
        covariance: np.ndarray,
    ) -> None:
        """Add the directed measurement-edge source_frame -> target_frame.

        This does not add the reverse edge automatically. If you have a valid
        reverse relative-pose covariance, add it separately.
        """
        covariance = symmetrize_covariance(covariance)

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
        """Find the minimum-determinant covariance path from source to target.

        For every node v, maintain:
            Sigma[v] = accumulated covariance of the best path found to v.

        Relaxation over directed-edge u -> v:
            Sigma_candidate = Sigma[u] + w(u, v)

        The candidate replaces Sigma[v] when:
            det(Sigma_candidate) < det(Sigma[v]).
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

        # Sigma[v] in the lecture slide.
        best_covariances: dict[int, np.ndarray] = {
            source_frame: zero_covariance,
        }

        # Scalar priority used only by heapq / Extract-Min.
        best_costs: dict[int, float] = {
            source_frame: 0.0,
        }

        # child -> (parent, covariance of the selected parent -> child edge)
        parents: dict[int, tuple[int, np.ndarray]] = {}

        # Counter prevents Python from trying to compare frame IDs in ties.
        insertion_counter = itertools.count()

        # Entries: (determinant score, tie breaker, frame ID).
        priority_queue: list[tuple[float, int, int]] = [
            (0.0, next(insertion_counter), source_frame),
        ]

        while priority_queue:
            current_cost, _, current_frame = heapq.heappop(priority_queue)

            # Ignore stale entries inserted before a later improvement.
            if current_cost > best_costs[current_frame]:
                continue

            if current_frame == target_frame:
                break

            current_covariance = best_covariances[current_frame]

            for edge in self.neighbors(current_frame):
                candidate_covariance = symmetrize_covariance(current_covariance + edge.covariance)
                candidate_cost = covariance_determinant(candidate_covariance)

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

        path_frame_ids = [target_frame]
        current_frame = target_frame

        while current_frame != source_frame:
            parent_frame, _ = parents[current_frame]
            path_frame_ids.append(parent_frame)
            current_frame = parent_frame

        path_frame_ids.reverse()

        return CovariancePath(
            frame_ids=path_frame_ids,
            covariance=best_covariances[target_frame],
            cost=best_costs[target_frame],
        )
