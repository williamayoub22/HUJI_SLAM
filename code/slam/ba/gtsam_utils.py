# gtsam_utils.py
# - convert_extrinsic_to_pose3(R, t)
# - make_stereo_calibration(calib)
# - make_stereo_camera(R, t, K_stereo)

import numpy as np
import gtsam

def make_gtsam_stereo_calibration(P1: np.ndarray, P2: np.ndarray) -> gtsam.Cal3_S2Stereo:
    fx = P1[0, 0]
    fy = P1[1, 1]

    skew = P1[0, 1]

    cx = P1[0, 2]
    cy = P1[1, 2]

    baseline = -P2[0, 3] / fx

    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)