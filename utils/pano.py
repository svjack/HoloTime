import numpy as np
import cv2

def GetPerspective(image, hFOV, wFOV, THETA, PHI, height, width):
    #
    # THETA is left/right angle, PHI is up/down angle
    # hFOV, wFOV, THETA, PHI are in degrees
    #
    equ_h = image.shape[0]
    equ_w = image.shape[1]
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))

    x_map = np.ones([height, width], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
    z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T

    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(D[:, :, np.newaxis], 3, axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(-THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2])
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])

    lon = lon.reshape([height, width]) / np.pi * 180
    lat = -lat.reshape([height, width]) / np.pi * 180

    lon_map = lon / 180 * equ_cx + equ_cx
    lat_map = lat / 90 * equ_cy + equ_cy

    persp_image = cv2.remap(image, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    return persp_image


def generate_point_cloud(image, depth):
    #
    # transform RGBD panorama to 3D point cloud
    #

    H = image.shape[0]
    W = image.shape[1]
    x = np.linspace(0, W, W)
    y = np.linspace(0, H, H)
    x, y = np.meshgrid(x, y)

    lon = -2 * np.pi * (x / W - 0.5)  # (-π to π)
    lat = -np.pi * (y / H - 0.5)  # (-π/2 to π/2)

    Z = depth * np.cos(lat) * np.cos(lon)
    X = depth * np.cos(lat) * np.sin(-lon)
    Y = depth * np.sin(-lat)

    point_cloud = np.array([X.flatten(), Y.flatten(), Z.flatten()]) # 3 x N
    point_color = (image.reshape(-1, 3).astype(np.float32)/255.0) # N x 3
    
    return point_cloud, point_color
