from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from ..io.calibration import read_calib
from ..geometry.correspondences import find_common_points
from ..features.detectors import extract_features, FeatureType
from ..io.image_loader import read_images
from ..features.matching import match_and_filter, get_matched_points
from ..geometry.ransac import ransac_pnp
from ..pipeline.stereo_pipeline import create_stereo_point_cloud
from ..geometry.transforms import to_homogeneous_transform


def track_sequence(
    sequence_dir: Path,
    num_frames: int | None = None,
    feature_type: FeatureType = "orb",
    num_features: int = 3000,
    use_ratio_test: bool = True,
    ratio_threshold: float = 0.75,
    deviation_threshold: float = 2.0,
    max_depth: float = 300.0,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """
    Tracks the left camera through the sequence.

    Returns:
        estimated_positions: Nx3 camera centers in left_0 coordinates.
        relative_transforms: list of relative transforms between consecutive frames.
        elapsed_time: tracking runtime in seconds.
    """
    P1, P2 = read_calib()
    K = P1[:, :3]

    total_frames = len(list((sequence_dir / "image_0").glob("*.png")))
    if num_frames is None:
        num_frames = total_frames
    else:
        num_frames = min(num_frames, total_frames)

    start_time = time.time()

    global_T = np.eye(4)
    estimated_positions = [np.zeros(3)]
    relative_transforms = []

    pc_prev = create_stereo_point_cloud(
        0,
        threshold=deviation_threshold,
        reject_negative_depth=True,
        max_depth=max_depth,
        feature_type=feature_type,
        num_features=num_features,
        use_ratio_test=use_ratio_test,
    )

    img_prev, _ = read_images(0)
    kp_prev, des_prev = extract_features(img_prev, feature_type, num_features)

    for frame_idx in tqdm(range(num_frames - 1), desc="Tracking frames"):
        curr_idx = frame_idx + 1

        img_curr, _ = read_images(curr_idx)
        kp_curr, des_curr = extract_features(img_curr, feature_type, num_features)

        pc_curr = create_stereo_point_cloud(
            curr_idx,
            threshold=deviation_threshold,
            reject_negative_depth=True,
            max_depth=max_depth,
            feature_type=feature_type,
            num_features=num_features,
            use_ratio_test=use_ratio_test,
        )

        if des_prev is None or des_curr is None:
            T_rel = np.eye(4)
        else:
            matches = match_and_filter(
                des_prev,
                des_curr,
                feature_type=feature_type,
                ratio=ratio_threshold,
            )

            if len(matches) < 4:
                T_rel = np.eye(4)
            else:
                pts_l0, pts_l1 = get_matched_points(kp_prev, kp_curr, matches)

                pts_3d, pts_l1_c, pts_l0_c, pts_r0_c, pts_r1_c = find_common_points(
                    pc_prev,
                    pc_curr,
                    pts_l0,
                    pts_l1,
                )

                if len(pts_3d) < 4:
                    T_rel = np.eye(4)
                else:
                    T_candidate, _ = ransac_pnp(
                        pts_3d,
                        pts_l1_c,
                        pts_l0_c,
                        pts_r0_c,
                        pts_r1_c,
                        K,
                        P1,
                        P2,
                    )

                    T_rel = T_candidate if T_candidate is not None else np.eye(4)

        if T_rel is None:
            T_rel = np.eye(4)

        step_T = to_homogeneous_transform(T_rel)

        relative_transforms.append(step_T)

        global_T = step_T @ global_T

        position = -global_T[:3, :3].T @ global_T[:3, 3]
        estimated_positions.append(position)

        pc_prev = pc_curr
        kp_prev = kp_curr
        des_prev = des_curr

    elapsed_time = time.time() - start_time

    return np.asarray(estimated_positions), relative_transforms, elapsed_time
