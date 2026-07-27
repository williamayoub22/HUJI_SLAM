import pickle
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
            "manager_poses": self.manager_poses
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
