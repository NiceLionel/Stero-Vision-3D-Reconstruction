import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane using intrinsics and extrinsics
    
    Hint:
    depth * corners = K @ T @ y, where y is the output world coordinates and T is the 4x4 matrix of Rt (3x4)

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points, here 2x2 correspinds to 4 corners
    """

    points = np.array(
        (
            (0, 0, 1),
            (width, 0, 1),
            (0, height, 1),
            (width, height, 1),
        ),
        dtype=np.float32,
    ).reshape(2, 2, 3)

    R = Rt[:, 0:3]
    T = Rt[:, -1]
    
    T_inv = -R.T @ T
    for i in range(2):
        for j in range(2):
            points[i, j, :] = (R.T @ (depth * (np.linalg.inv(K) @ points[i, j, :])) + T_inv)

    return points


def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    
    Hint:
    Z * projections = K @ T @ p, where p is the input points and projections is the output, T is the 4x4 matrix of Rt (3x4)
    
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    R = Rt[:, 0:3]
    T = Rt[:, -1]
    height = points.shape[0]
    width = points.shape[1]
    projections = np.zeros((height, width, 2))
    for i in range(height):
        for j in range(width):
            temp = K @ (R @ points[i, j, :] + T)
            temp = temp / temp[-1]
            projections[i, j, :] = temp[0:2]
    return projections


def warp_neighbor_to_ref(
    backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor
):
    """
    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]
    
    point = np.array(((0, 0),(width, 0),(0, height),(width, height)), dtype=np.float32)
    corner_point_backprojected = backproject_fn(K_ref, width, height, depth, Rt_ref)
    projection = project_fn(K_neighbor, Rt_neighbor, corner_point_backprojected)
    projected_points = np.zeros((4, 2))
    projected_points[0] = projection[0, 0, :]
    projected_points[1] = projection[0, 1, :]
    projected_points[2] = projection[1, 0, :]
    projected_points[3] = projection[1, 1, :]

    Homography, _ = cv2.findHomography(projected_points, point)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, Homography, dsize=(width, height))
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """
    Compute the cost map between src and dst patchified images via the ZNCC metric

    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value,
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    height = src.shape[0]
    width = src.shape[1]
    K2 = src.shape[2]

    
    zncc_3_dimension = np.zeros((height, width, 3))

    for i in range(3):
        single_src = src[:, :, :, i].transpose(2, 0, 1).reshape(K2, -1)
        single_dst = dst[:, :, :, i].transpose(2, 0, 1).reshape(K2, -1)
        src_W = np.mean(single_src, axis=0)
        dst_W = np.mean(single_dst, axis=0)
        src_sigma = np.std(single_src, axis=0)
        dst_sigma = np.std(single_dst, axis=0)
        num = (single_src - src_W) * (single_dst - dst_W)
        zncc_3_dimension[:, :, i] = (np.sum(num, axis=0) / ((dst_sigma + EPS) * (src_sigma + EPS))).reshape(height, width)                                                                                                         
    zncc = np.sum(zncc_3_dimension, axis=2)
    
    return zncc  # height x width


def backproject(dep_map, K):
    """
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    height = dep_map.shape[0]
    width = dep_map.shape[1]
    dep_map_flatted = dep_map.flatten()

    point_value = np.zeros((3, height * width))
    point_value[2, :] = 1
    point_value[1, :] = _v.flatten()
    point_value[0, :] = _u.flatten()

    cam_point = np.zeros((3, height * width))
    xyz_cam = np.zeros((height, width, 3))

    for i in range(height * width):
        cam_point[:, i] = dep_map_flatted[i] * (np.linalg.inv(K) @ point_value[:, i])
        u = int(point_value[0, i])
        v = int(point_value[1, i])

        xyz_cam[v, u, 0] = cam_point[:, i][0]
        xyz_cam[v, u, 1] = cam_point[:, i][1]
        xyz_cam[v, u, 2] = cam_point[:, i][2]

    return xyz_cam
