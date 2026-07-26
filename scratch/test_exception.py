import sys
sys.path.append('.')
from src.ex8 import load_or_build_db, make_stereo_camera, read_stereo_calibration, make_gtsam_stereo_calibration, optimize_track_landmark, get_stereo_observation, stereo_point_from_triplet, stereo_image_distances
from src.slam.geometry.transforms import to_homogeneous_transform
import numpy as np
from src.slam.config import GLOBAL_CAMERA_MATRICES_PATH
import traceback
import gtsam

def main():
    db = load_or_build_db()
    pnp_extrinsics = np.asarray(np.load(GLOBAL_CAMERA_MATRICES_PATH), dtype=float)
    world_to_camera_by_frame = {
        frame_id: to_homogeneous_transform(extrinsic)
        for frame_id, extrinsic in enumerate(pnp_extrinsics)
    }
    
    P1, P2 = read_stereo_calibration()
    calibration = make_gtsam_stereo_calibration(P1, P2)
    local_stereo_camera = make_stereo_camera(np.eye(4, dtype=float), calibration)
    
    manager_2d = db.manager_2d
    for track_id in manager_2d.all_tracks():
        track_frames = sorted(manager_2d.frames(track_id))
        usable_frames = [f for f in track_frames if f in world_to_camera_by_frame]
        if len(usable_frames) < 2: continue
        
        triangulation_frame = usable_frames[-1]
        evaluation_frames = [f for f in usable_frames if 0 <= triangulation_frame - f <= 50]
        if len(evaluation_frames) < 2: continue
        
        try:
            landmark_world = optimize_track_landmark(
                manager_2d=manager_2d,
                track_id=track_id,
                frame_ids=[triangulation_frame],
                world_to_camera_by_frame=world_to_camera_by_frame,
                calibration=calibration,
                measurement_sigma_pixels=1.0,
            )
        except Exception as e:
            print("Error optimizing track_id", track_id, ":", repr(e))
            traceback.print_exc()
            return
            
        landmark_homogeneous = np.array([*landmark_world, 1.0])
        
        for frame_id in evaluation_frames[:-1]:
            try:
                frame_from_triangulation = np.eye(4)
                landmark_in_frame_camera = landmark_homogeneous[:3]
                projection = local_stereo_camera.project(gtsam.Point3(landmark_in_frame_camera))
                
                measurement = stereo_point_from_triplet(
                    get_stereo_observation(manager_2d, frame_id, track_id)
                )
                
                left_error, right_error = stereo_image_distances(
                    measurement=measurement,
                    projection=projection,
                )
            except Exception as e:
                print("Error projecting track_id", track_id, ":", repr(e))
                traceback.print_exc()
                return
        print("Success on track_id", track_id)
        return

if __name__ == "__main__":
    main()
