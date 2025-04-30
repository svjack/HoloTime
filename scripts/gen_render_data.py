import cv2
import numpy as np
import os
import open3d as o3d
from plyfile import PlyData, PlyElement
import argparse
import shutil
import math
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.pano import generate_point_cloud
from utils.trajectory import get_pcdGenPoses
from Depth_estimation.utils.geo_utils import panorama_to_pers_cameras
from image_warper.utils import gt_warping


def generate_test_data(trajectory, save_dir, args):
    output_dir = os.path.join(save_dir, trajectory)
    os.makedirs(output_dir, exist_ok=True)

    fov_x = np.radians(args.fov_x)
    fov_y = np.radians(args.fov_y)
    width = args.pers_res
    height = args.pers_res
    focal_x = width / (2 * np.tan(fov_x / 2))
    focal_y = height / (2 * np.tan(fov_y / 2))

    render_poses = get_pcdGenPoses(trajectory)
    for idx in range(len(render_poses)):
        os.makedirs(os.path.join(output_dir, f'cam{idx:03d}'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'cam{idx:03d}', 'frames'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'cam{idx:03d}', 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'cam{idx:03d}', 'times'), exist_ok=True)
        
        cv2.imwrite(os.path.join(output_dir, f'cam{idx:03d}', 'frames', f'{0:05d}.png'), np.zeros((height, width, 3), dtype=np.uint8))
        cv2.imwrite(os.path.join(output_dir, f'cam{idx:03d}', 'masks', f'{0:05d}.png'), np.ones((height, width), dtype=np.uint8))
        with open(os.path.join(output_dir, f'cam{idx:03d}', 'times', f'{0:05d}.txt'), 'w') as f:
            f.write(str(math.sin(idx / 30) / 2 + 1 / 2))
        np.save(os.path.join(output_dir, f'cam{idx:03d}', f'traj.npy'), render_poses[idx])

    # 保存相机参数
    with open(os.path.join(output_dir, 'camera_params.txt'), 'w') as f:
        f.write(f'fov_x: {fov_x}\nfov_y: {fov_y}\n')
        f.write(f'focal_x: {focal_x}\nfocal_y: {focal_y}\n')
        f.write(f'num_cams: {len(render_poses)}\n')
        f.write(f'frame_count: {1}\n')
        f.write(f'height: {height}\n')
        f.write(f'width: {width}\n')

    # 创建 points3D_multipleview.ply 空文件
    with open(os.path.join(output_dir, 'points3D.ply'), 'w') as ply_file:
        pass  # 创建一个空文件

    return


if __name__ == "__main__":
    # 用命令行解析name
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=str, default='rotate360', choices=['rotate360', 'rotateshow', 'back_and_forth', 'headbanging', 'llff'], help='trajectory for render')
    parser.add_argument('--save_dir', type=str, default='data', help='trajectory for render')

    parser.add_argument('--pers_res', type=int, default=512, help='Resolution of the perspective images')
    parser.add_argument('--fov_x', type=float, default=75, help='Field of view ratio (degrees)')
    parser.add_argument('--fov_y', type=float, default=75, help='Field of view ratio (degrees)')

    args = parser.parse_args()
    trajectory = args.trajectory
    save_dir = args.save_dir
    generate_test_data(trajectory, save_dir, args)
