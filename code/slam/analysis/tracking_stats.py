from dataclasses import dataclass
from typing import List

import numpy as np

from ..tracking_database import TrackingDB



@dataclass
class TrackingStats:
    total_tracks: int
    num_frames: int
    mean_track_length: float
    max_track_length: int
    min_track_length: int
    mean_frame_links: float

    def __str__(self) -> str:
        return (
            "Tracking statistics:\n"
            f"  Total number of tracks: {self.total_tracks}\n"
            f"  Number of frames: {self.num_frames}\n"
            f"  Mean track length: {self.mean_track_length:.2f}\n"
            f"  Max track length: {self.max_track_length}\n"
            f"  Min track length: {self.min_track_length}\n"
            f"  Mean number of frame links: {self.mean_frame_links:.2f}"
        )

    def to_latex_table_rows(self) -> str:
        return (
            f"Total number of tracks & {self.total_tracks} \\\\\n"
            f"Number of frames & {self.num_frames} \\\\\n"
            f"Mean track length & {self.mean_track_length:.2f} \\\\\n"
            f"Maximum track length & {self.max_track_length} \\\\\n"
            f"Minimum track length & {self.min_track_length} \\\\\n"
            f"Mean number of frame links & {self.mean_frame_links:.2f} \\\\"
        )


""" all track lengths (excluding trivial length-1 tracks per exercise spec) """
def get_track_lengths(db: TrackingDB, min_length: int = 2) -> List[int]:
    return [
        db.track_length(track_id)
        for track_id in db.all_tracks()
        if db.track_length(track_id) >= min_length
    ]


""" number of non-trivial tracks appearing in each frame """
def get_frame_link_counts(db: TrackingDB) -> List[int]:
    return [
        len(db.tracks(frame_id))
        for frame_id in db.all_frames()
    ]


""" compute tracking statistics required in exercise 4.2 """
def compute_tracking_statistics(db: TrackingDB) -> TrackingStats:
    track_lengths = get_track_lengths(db)
    frame_link_counts = get_frame_link_counts(db)

    if len(track_lengths) == 0:
        return TrackingStats(
            total_tracks=0,
            num_frames=db.frame_num(),
            mean_track_length=0.0,
            max_track_length=0,
            min_track_length=0,
            mean_frame_links=0.0,
        )

    return TrackingStats(
        total_tracks=len(track_lengths),
        num_frames=db.frame_num(),
        mean_track_length=float(np.mean(track_lengths)),
        max_track_length=int(np.max(track_lengths)),
        min_track_length=int(np.min(track_lengths)),
        mean_frame_links=float(np.mean(frame_link_counts)),
    )

