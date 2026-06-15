# ex5.py
# - choose_track_of_length(db, min_length=10)
# - get_track_stereo_measurements(db, track_id)
# - run_reprojection_test_for_track(...)

import numpy as np

from slam.pipeline.tracking_pipeline import to_homogeneous_transform
from slam.config import PROJECT_DIR
from slam.tracking_database import TrackingDB
from slam.io.calibration import read_calib
from slam.ba.gtsam_utils import make_gtsam_stereo_calibration
from slam.pipeline.database_pipeline import load_tracking_database

from ex4 import DB_PATH
RELATIVE_TRANSFORMS_PATH = PROJECT_DIR / "outputs" / "ex3" / "relative_transforms.npy"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ex5"

def load_tracking_db() -> TrackingDB:
    db = TrackingDB()
    db.load(str(DB_PATH))
    return db


def load_relative_transforms() -> list[np.ndarray]:
    if not RELATIVE_TRANSFORMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RELATIVE_TRANSFORMS_PATH}. Run ex3.py first."
        )

    arr = np.load(RELATIVE_TRANSFORMS_PATH, allow_pickle=True)
    return [to_homogeneous_transform(np.asarray(T)) for T in arr]

def compose_global_camera_matrices(relative_transforms: list[np.ndarray]) -> list[np.ndarray]:
    global_matrices = [np.eye(4)]
    T_global = np.eye(4)

    for T_prev_to_cur in relative_transforms:
        T_global = T_prev_to_cur @ T_global
        global_matrices.append(T_global.copy())

    return global_matrices

def section_5_1():
    db = load_tracking_db()
    relative_transforms = load_relative_transforms()
    camera_matrices = compose_global_camera_matrices(relative_transforms)
    P1, P2 = read_calib()

    K = make_gtsam_stereo_calibration(P1, P2)

    print(f"Loaded DB with {db.frame_num()} frames and {db.track_num()} tracks")
    print(f"Loaded {len(camera_matrices)} global camera matrices")
    print("GTSAM stereo calibration:")

    print(K)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    section_5_1()