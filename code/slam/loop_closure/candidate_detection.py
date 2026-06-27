# loop over (c_i, c_n), rank candidates

from dataclasses import dataclass

import numpy as np
from slam.pose_graph.covariance_routing import (
    CovarianceGraph,
)


@dataclass(frozen=True)
class LoopClosureCandidate:
    frame_i: int
    frame_n: int
    relative_covariance: np.ndarray
    uncertainty_score: float
    path_keyframes: list[int]


def find_loop_closure_candidates(
    keyframe_ids: list[int],
    covariance_graph: CovarianceGraph,
    min_keyframe_index_gap: int,
) -> list[LoopClosureCandidate]:
    candidates = []

    for n_index, frame_n in enumerate(keyframe_ids):
        for i_index, frame_i in enumerate(keyframe_ids[:n_index]):
            # Avoid trivial nearby-frame matches.
            if n_index - i_index < min_keyframe_index_gap:
                continue

            path = covariance_graph.shortest_path(
                source=frame_i,
                target=frame_n,
            )

            if path is None:
                continue

            candidates.append(
                LoopClosureCandidate(
                    frame_i=frame_i,
                    frame_n=frame_n,
                    relative_covariance=path.covariance,
                    uncertainty_score=path.cost,
                    path_keyframes=path.keyframes,
                )
            )

    return candidates


# For now, this function can return all candidate pairs. Later you can add:
#
# * max_candidates_per_frame;
# * maximum allowed uncertainty;
# * top-k candidates by uncertainty;
# * spatial proximity checks;
# * visual matching and PnP verification.
