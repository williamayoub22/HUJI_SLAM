
def find_common_points(pc0, pc1, l0_pts, l1_pts):
    """
    Finds points existing in both stereo point clouds and temporal matches.

    Args:
        pc0: StereoPointCloud for pair 0
        pc1: StereoPointCloud for pair 1
        l0_pts: Nx2 temporal match points on previous left image (left_0)
        l1_pts: Nx2 temporal match points on current left image (left_1)

    Returns:
        pts_3d: common 3D points from stereo pair 0 triangulation
        pts_l1: corresponding 2D points on left_1
        pts_l0: corresponding 2D points on left_0
        pts_r0: corresponding 2D points on right_0
        pts_r1: corresponding 2D points on right_1
    """
    # Map 2D point locations to their index in the respective point clouds
    stereo0_dict = {(float(pt[0]), float(pt[1])): i for i, pt in enumerate(pc0.left_inliers)}
    stereo1_dict = {(float(pt[0]), float(pt[1])): i for i, pt in enumerate(pc1.left_inliers)}

    idx_temporal, idx_stereo0, idx_stereo1 = [], [], []
    for i, (pt0, pt1) in enumerate(zip(l0_pts, l1_pts)):
        key0 = (float(pt0[0]), float(pt0[1]))
        key1 = (float(pt1[0]), float(pt1[1]))

        # Keep only key-points matched on all four images
        if key0 in stereo0_dict and key1 in stereo1_dict:
            idx_temporal.append(i)
            idx_stereo0.append(stereo0_dict[key0])
            idx_stereo1.append(stereo1_dict[key1])

    return (pc0.points_3d[idx_stereo0],
            l1_pts[idx_temporal],
            l0_pts[idx_temporal],
            pc0.right_inliers[idx_stereo0],
            pc1.right_inliers[idx_stereo1])
