"""
Unified Tracking Database Interface.

This module provides the `SlamDatabase` and `TrackManager2D` facades which wrap the low-level 
underlying tracking database implementation. It provides clean, high-level API methods for:
- Querying track lengths and visibility.
- Fetching specific (x_left, x_right, y) stereo observations.
- Navigating the bipartite graph between Tracks and Frames.
"""

import pickle
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.slam.geometry.triangulation import triangulate_opencv

from .tracking_db import TrackingDB


class Manager3D:
    def __init__(self):
        self._points = {}

    def add_point(self, track_id: int, point):
        self._points[track_id] = point

    def get_point(self, track_id: int):
        return self._points[track_id]

    def get_all_points(self):
        return self._points


class ManagerPoses:
    def __init__(self):
        self._poses = {}

    def add_pose(self, frame_id: int, pose):
        self._poses[frame_id] = pose

    def get_pose(self, frame_id: int):
        return self._poses[frame_id]

    def get_all_poses(self):
        return self._poses


class SlamDatabase:
    def __init__(self):
        self.manager_2d = TrackingDB()
        self.manager_3d = Manager3D()
        self.manager_poses = ManagerPoses()

    def serialize(self, base_filename: str):
        data = {
            "manager_2d": self.manager_2d,
            "manager_3d": self.manager_3d,
            "manager_poses": self.manager_poses,
        }
        filename = base_filename + ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(data, file)
        print("SlamDatabase serialized to", filename)

    @classmethod
    def load(cls, base_filename: str) -> "SlamDatabase":
        filename = base_filename + ".pkl"
        with open(filename, "rb") as file:
            data = pickle.load(file)
            db = cls()
            db.manager_2d = data["manager_2d"]
            db.manager_3d = data["manager_3d"]
            db.manager_poses = data["manager_poses"]
            print("SlamDatabase loaded from", filename)
            return db

    def triangulate_all_tracks(self, P1: np.ndarray, P2: np.ndarray):
        """Triangulate all tracks using their initialization frame."""
        print("Triangulating 3D points...")
        for track_id in tqdm(self.manager_2d.all_tracks(), desc="Triangulating points"):
            track_frames = self.manager_2d.frames(track_id)
            if len(track_frames) < 2:
                continue
            initialization_frame = track_frames[-1]
            T_world_to_camera = self.manager_poses.get_pose(initialization_frame)
            xl, xr, y = self.manager_2d.link_triplet(initialization_frame, track_id)
            left_pt = np.array([[xl, y]])
            right_pt = np.array([[xr, y]])
            point_3d_camera = triangulate_opencv(P1, P2, left_pt, right_pt)[0]
            T_camera_to_world = np.linalg.inv(T_world_to_camera)
            point_4d = np.append(point_3d_camera, 1.0)
            point_3d_world = (T_camera_to_world @ point_4d)[:3]
            self.manager_3d.add_point(track_id, point_3d_world)
