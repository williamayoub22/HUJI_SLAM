import numpy as np

class Manager3D:
    def __init__(self):
        self.points = {}

    def add_point(self, track_id: int, point: np.ndarray):
        self.points[track_id] = point

    def get_point(self, track_id: int) -> np.ndarray | None:
        return self.points.get(track_id)
        
    def all_points(self) -> dict[int, np.ndarray]:
        return self.points
