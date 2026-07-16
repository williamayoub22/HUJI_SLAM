import numpy as np

class ManagerPoses:
    def __init__(self):
        self.poses = {}

    def add_pose(self, frame_id: int, pose: np.ndarray):
        self.poses[frame_id] = pose

    def get_pose(self, frame_id: int) -> np.ndarray | None:
        return self.poses.get(frame_id)
        
    def get_all_poses(self) -> dict[int, np.ndarray]:
        return self.poses
