#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import cv2
import os
import sys
from PIL import Image
from typing import NamedTuple
from tqdm import tqdm
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
import glob
import natsort
from simple_knn._C import distCUDA2
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float
    mask: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    nerf_normalization: dict
    ply_path: str
    duration: int
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    times = np.vstack([vertices['t']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    number = 4
    old_times = times
    old_positions = positions
    old_colors = colors
    old_normals = normals

    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)

def storePly(path, xyzt, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('t','f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readMultipleViewinfos(datadir):
    cameras_intrinsic_file = os.path.join(datadir, "camera_params.txt")
    cam_extrinsics = read_extrinsics(datadir)
    focal1, focal2, width, height, cams_num, frame_count = read_intrinsics(cameras_intrinsic_file)
    #focal2 = None

    from scene.multipleview_dataset import multipleview_dataset
    train_cam_infos_ = multipleview_dataset(cam_extrinsics, datadir, frame_count, focal1, focal2)
    near = 0.01
    far = 100
    train_cam_infos = format_infos(train_cam_infos_, near=near, far=far, cams_num=cams_num)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(datadir, "points3D.ply")
    bin_path = os.path.join(datadir, "points3D.bin")
    txt_path = os.path.join(datadir, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        print("Loading point cloud...")
        pcd = fetchPly(ply_path)
        
    except Exception as e:
        print("Could not load ply file, error: ", e)
        pcd = None
    
    #print(pcd)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           duration=frame_count)
    return scene_info

def read_intrinsics(intrinsics_file):
    with open(intrinsics_file) as f:
        lines = f.readlines()
    for line in lines:
        if "focal_x" in line:
            focal1 = [float(data) for data in line.split(" ")[1:]]
        if "focal_y" in line:
            focal2 = [float(data) for data in line.split(" ")[1:]]
        if "width" in line:
            width = [int(data) for data in line.split(" ")[1:]]
        if "height" in line:
            height = [int(data) for data in line.split(" ")[1:]]
        if "num_cams" in line:
            num_cams = [int(data) for data in line.split(" ")[1:]]
        if "frame_count" in line:
            frame_count = [int(data) for data in line.split(" ")[1:]]
    #print(focal1, focal2, width, height, num_cams)
    return focal1, focal2, width, height, num_cams, frame_count[0]

def read_extrinsics(data_dir):
    # cam000, cam001, ...
    cam_extrinsics = []
    files = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for cam_folder in sorted(files, key=lambda x: int(x[3:])):
        # read traj
        traj_file = os.path.join(data_dir, cam_folder, "traj.npy")
        traj = np.load(traj_file)
        cam_extrinsics.append(traj)
    return cam_extrinsics

def format_infos(dataset, near, far, cams_num):
    # loading
    cameras = []
    cxr = 0
    cyr = 0
    cams_num = [sum(cams_num[:i+1]) for i in range(len(cams_num))]

    for idx in tqdm(range(len(dataset))):
        image_path = dataset.image_paths[idx]
        image_name = dataset.image_names[idx]
        camera_name = image_name.split("_")[0]
        image = Image.open(image_path)
        cam_type = 0
        total_num = 0
        for i, num in enumerate(cams_num):
            if int(camera_name) < num:
                cam_type = i
                break
        
        mask_path = image_path.replace("frames", "masks")
        mask = Image.open(mask_path)
        mask = (np.array(mask) / 255.0).astype(np.uint8)

        time = dataset.image_times[idx]
        R,T = dataset.load_pose(idx)
        FovX = focal2fov(dataset.focal1[cam_type], image.size[0])
        FovY = focal2fov(dataset.focal2[cam_type], image.size[1])

        hpdirecitons = 1
        pose = 1
        
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=camera_name, width=image.size[0], height=image.size[1],
                            near=near, far=far, timestamp=time, pose=pose, hpdirecitons=hpdirecitons, cxr=cxr, cyr=cyr, mask=mask))

    return cameras

sceneLoadTypeCallbacks = {
    "Panorama": readMultipleViewinfos,
}


