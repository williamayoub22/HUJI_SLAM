import pickle
from .manager_2d import Manager2D
from .manager_3d import Manager3D
from .manager_poses import ManagerPoses

class SlamDatabase:
    def __init__(self):
        self.manager_2d = Manager2D()
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
