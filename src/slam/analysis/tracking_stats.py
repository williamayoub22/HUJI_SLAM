"""Utilities for computing summary statistics of the tracking database."""

from dataclasses import dataclass

import numpy as np

from ..tracking_database import TrackingDB


@dataclass
class TrackingStats:
    """Summary statistics describing tracks and frame-level associations.

    Attributes:
        total_tracks: Number of non-trivial tracks.
        num_frames: Number of frames in the tracking database.
        mean_track_length: Mean number of observations per track.
        max_track_length: Maximum track length.
        min_track_length: Minimum track length.
        mean_frame_links: Mean number of tracks associated with a frame.
    """

    total_tracks: int
    num_frames: int
    mean_track_length: float
    max_track_length: int
    min_track_length: int
    mean_frame_links: float

    def __str__(self) -> str:
        """Return a readable multi-line summary of the tracking statistics."""
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
        """Return the statistics as LaTex table rows."""
        return (
            f"Total number of tracks & {self.total_tracks} \\\\\n"
            f"Number of frames & {self.num_frames} \\\\\n"
            f"Mean track length & {self.mean_track_length:.2f} \\\\\n"
            f"Maximum track length & {self.max_track_length} \\\\\n"
            f"Minimum track length & {self.min_track_length} \\\\\n"
            f"Mean number of frame links & {self.mean_frame_links:.2f} \\\\"
        )


def get_track_lengths(db: TrackingDB, min_length: int = 2) -> list[int]:
    """Return lengths of tracks with at least ``min_length`` observations."""
    return [
        db.track_length(track_id)
        for track_id in db.all_tracks()
        if db.track_length(track_id) >= min_length
    ]


def get_frame_link_counts(db: TrackingDB) -> list[int]:
    """Return the number of tracks associated with each frame."""
    return [len(db.tracks(frame_id)) for frame_id in db.all_frames()]


def compute_tracking_statistics(db: TrackingDB) -> TrackingStats:
    """Compute the tracking statistics required for Exercise 4.2."""
    track_lengths = get_track_lengths(db)
    frame_link_counts = get_frame_link_counts(db)

    if not track_lengths:
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
