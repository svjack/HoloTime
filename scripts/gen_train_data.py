import torch
import os
import cv2
import numpy as np
import random
import argparse
import open3d as o3d
from plyfile import PlyData, PlyElement
import shutil
import torch.nn.functional as F
from decord import VideoReader, cpu
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.pano import generate_point_cloud
from utils.trajectory import generate_seed_perturb
from utils.inference_utils import save_video
from utils.camera import CameraParams
from image_warper.utils_gpu import gt_warping

from PanoFlowAPI.video2flow import abstract_flow
from Depth_estimation.utils.geo_utils import panorama_to_pers_cameras, panorama_to_pers_directions
from Depth_estimation.utils.camera_utils import *
from Depth_estimation.geo_predictors.pano_geo_predictor_last import *


def generate_temporal_point_cloud(frame_list, depth_list, mask_pre_list, mask_post_list, output_dir, args):
    #
    # transform RGBD panorama video to temporal 3D point cloud
    #
    
    device=args.device
    gray_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frame_list])
    std = np.std(gray_frames, axis=0)
    tex_mask  = (std > args.tex_threshold)
    kernel = np.ones((5, 5), np.uint8)
    tex_mask = cv2.dilate(tex_mask.astype(np.uint8), kernel, iterations=1)
    
    flow_masks_pre = np.array(mask_pre_list)
    flow_masks_post = np.array(mask_post_list)
    flow_masks = (flow_masks_pre | flow_masks_post)

    frame_width = frame_list[0].shape[1]
    frame_height = frame_list[0].shape[0]
    
    total_point_cloud = []
    total_point_color = []
    times = []
    len_frames = len(frame_list)

    ############### Downsampling to reduce the scene scale and boost training speed ###############
    ############### Resize frame resolution to get ideal point cloud ###############
    for i in range(len_frames):
        frame = frame_list[i]
        depth = depth_list[i]
        if i % 2 == 0:
            frame = cv2.resize(frame, (frame_width // 8, frame_height // 8))
            depth = cv2.resize(depth[...,0], (frame_width // 8, frame_height // 8))
            point_cloud, point_color = generate_point_cloud(frame, depth)
        else:
            frame = cv2.resize(frame, (frame_width // 4, frame_height // 4))
            depth = cv2.resize(depth[...,0], (frame_width // 4, frame_height // 4))
            point_cloud, point_color = generate_point_cloud(frame, depth)

            ### motion part ###
            flow_mask = flow_masks[i-1]
            if i < len_frames-1:
                pre_mask = flow_masks_pre[i]
                flow_mask = flow_mask | pre_mask
            mask = (flow_mask | tex_mask)
            mask = cv2.resize(mask, (frame_width // 4, frame_height // 4)).reshape(-1)

            point_cloud = point_cloud[:, mask == 1]
            point_color = point_color[mask == 1]
            # random perturbation
            point_cloud += np.random.rand(*point_cloud.shape) * 0.02 - 0.01
 
        total_point_cloud.append(point_cloud)
        total_point_color.append(point_color)
        times.append(np.ones(point_cloud.shape[1]) * i / len_frames)
    
    point_cloud = np.concatenate(total_point_cloud, axis=1)
    point_color = np.concatenate(total_point_color, axis=0)
    normals = np.zeros_like(point_cloud)
    times = np.concatenate(times)

    vertex = np.zeros(point_cloud.shape[1], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('t','f4'),
                                                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                                ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
    vertex['x'] = point_cloud[0]
    vertex['y'] = point_cloud[1]
    vertex['z'] = point_cloud[2]
    vertex['red'] = point_color[:, 0] * 255
    vertex['green'] = point_color[:, 1] * 255
    vertex['blue'] = point_color[:, 2] * 255
    vertex['nx'] = normals[0]
    vertex['ny'] = normals[1]
    vertex['nz'] = normals[2]
    vertex['t'] = times
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(os.path.join(output_dir, 'points3D.ply'))

    print("Point cloud Num.", point_cloud.shape[1])


def gen_train_data(frame_list, depth_list, mask_pre_list, mask_post_list, output_dir, args):
    # frame_list: [H, W, 3] x N
    # depth_list: [H, W, 1] x N
    # mask_pre_list: [H, W] x (N-1)
    # mask_post_list: [H, W] x (N-1)
    
    device = args.device
    c2w_list = []
    pers_res = args.pers_res
    height = pers_res
    width = pers_res

    ratio_s = args.fov_ratio_s
    ratio_l = args.fov_ratio_l
    perturb = args.perturb

    # Cameras with small ratio
    cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(gen_res=pers_res, ratio=ratio_s)
    pers_num = len(cur_pers_dirs)
    img_coords = direction_to_img_coord(cur_pers_dirs)
    sample_coords_s = img_coord_to_sample_coord(img_coords)
    c2ws_s, fov_x_s, fov_y_s = panorama_to_pers_cameras(ratio=ratio_s)
    c2w_list.append(c2ws_s.cpu().numpy())
    fov_x_s = fov_x_s[0].item()
    fov_y_s = fov_y_s[1].item()
  
    # Cameras with large ratio
    cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(gen_res=pers_res, ratio=ratio_l)
    img_coords = direction_to_img_coord(cur_pers_dirs)
    sample_coords_l = img_coord_to_sample_coord(img_coords)
    c2ws_l, fov_x_l, fov_y_l = panorama_to_pers_cameras(ratio=ratio_l)
    c2w_list.append(c2ws_l.cpu().numpy())
    fov_x_l = fov_x_l[0].item()
    fov_y_l = fov_y_l[1].item()

    c2ws = np.concatenate(c2w_list, axis=0)
    len_frames = len(frame_list)
    camera_params_s = CameraParams(width, height, fov_x_s, fov_y_s)
    camera_params_l = CameraParams(width, height, fov_x_l, fov_y_l)

    intrinsic_s = (camera_params_s.focal[0], camera_params_s.focal[1], camera_params_s.cx, camera_params_s.cy)

    trajs = generate_seed_perturb(perturb)
    trajs = torch.from_numpy(trajs).to(device)
    traj_num = trajs.shape[0]
    
    
    for i in range(len_frames):
        print('Processing frame:', i)
        frame = frame_list[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        depth = depth_list[i]
        frame = torch.from_numpy(frame).permute(2, 0, 1).float().cuda()
        depth = torch.from_numpy(depth).permute(2, 0, 1).float().cuda()

        pers_imgs = F.grid_sample(frame[None].expand(20, -1, -1, -1), sample_coords_s, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pers_depths = F.grid_sample(depth[None].expand(20, -1, -1, -1), sample_coords_s, padding_mode='border') # [n_pers, 1, gen_res, gen_res]
        pers_imgs_l = F.grid_sample(frame[None].expand(20, -1, -1, -1), sample_coords_l, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pers_depths_l = F.grid_sample(depth[None].expand(20, -1, -1, -1), sample_coords_l, padding_mode='border') # [n_pers, 1, gen_res, gen_res]

        ############# Cameras with small ratio #############
        for p in range(pers_num):
            idx = p * traj_num
            c2w = c2ws[p]
            c2w_ = np.zeros((4, 4))
            c2w_[:3, :3] = c2w[:3, :3]
            c2w_[3, 3] = 1.0
            ### make dirs
            if i == 0:
                os.makedirs(os.path.join(output_dir, f'cam{idx:03d}'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, f'cam{idx:03d}', 'frames'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, f'cam{idx:03d}', 'masks'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, f'cam{idx:03d}', 'times'), exist_ok=True)
            
            persp_frame = pers_imgs[p].permute(1, 2, 0)#.cpu().numpy()
            pers_depth = pers_depths[p].permute(1, 2, 0)[..., 0]#.cpu().numpy()[..., 0]
            rgbs_warp_temp, masks_warp_temp, depth_warp_temp = gt_warping(persp_frame, pers_depth, trajs[0], trajs[1:], height, width,
                                                                        #   logpath=os.path.join(save_path_warp, 'rgbs_support/%05d_%03d'%(N_iter, ii)), 
                                                            intrinsic=intrinsic_s, warp_depth=True, bilinear_splat=True)
            persp_frame = persp_frame.cpu().numpy()
            cv2.imwrite(os.path.join(output_dir, f'cam{idx:03d}', 'frames', f'{i:05d}.png'), persp_frame)
            cv2.imwrite(os.path.join(output_dir, f'cam{idx:03d}', 'masks', f'{i:05d}.png'), (np.ones_like(persp_frame[..., 0]))*255)
            np.save(os.path.join(output_dir, f'cam{idx:03d}', f'traj.npy'), c2w_[:3, :])
            with open(os.path.join(output_dir, f'cam{idx:03d}', 'times', f'{i:05d}.txt'), 'w') as f:
                f.write(str(i/(len_frames-1)))

            for index in range(traj_num-1):
                ### make dirs
                if i == 0:
                    os.makedirs(os.path.join(output_dir, f'cam{idx+index+1:03d}'), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, f'cam{idx+index+1:03d}', 'frames'), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, f'cam{idx+index+1:03d}', 'masks'), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, f'cam{idx+index+1:03d}', 'times'), exist_ok=True)

                cv2.imwrite(os.path.join(output_dir, f'cam{idx+index+1:03d}', 'frames', f'{i:05d}.png'), rgbs_warp_temp[index].astype(np.uint8))
                cv2.imwrite(os.path.join(output_dir, f'cam{idx+index+1:03d}', 'masks', f'{i:05d}.png'), masks_warp_temp[index]*255)
                with open(os.path.join(output_dir, f'cam{idx+index+1:03d}', 'times', f'{i:05d}.txt'), 'w') as f:
                    f.write(str(i/(len_frames-1)))
                c2w_perturb = c2w_
                if index == 0:
                    c2w_perturb[:3, 3:4] = np.array([perturb, 0, 0]).reshape(3,1)
                    np.save(os.path.join(output_dir, f'cam{idx+index+1:03d}', f'traj.npy'), c2w_perturb[:3, :])
                elif index == 1:
                    c2w_perturb[:3, 3:4] = np.array([0, perturb, 0]).reshape(3,1)
                    np.save(os.path.join(output_dir, f'cam{idx+index+1:03d}', f'traj.npy'), c2w_perturb[:3, :])
                elif index == 2:
                    c2w_perturb[:3, 3:4] = np.array([-perturb, 0, 0]).reshape(3,1)
                    np.save(os.path.join(output_dir, f'cam{idx+index+1:03d}', f'traj.npy'), c2w_perturb[:3, :])
                elif index == 3:
                    c2w_perturb[:3, 3:4] = np.array([0, -perturb, 0]).reshape(3,1)
                    np.save(os.path.join(output_dir, f'cam{idx+index+1:03d}', f'traj.npy'), c2w_perturb[:3, :])

        ############# Cameras with large ratio #############
        for p in range(pers_num):
            idx = p + pers_num * traj_num
            c2w = c2ws[pers_num+p]
            c2w_ = np.zeros((4, 4))
            c2w_[:3, :3] = c2w[:3, :3]
            c2w_[3, 3] = 1.0
            if i == 0:
                os.makedirs(os.path.join(output_dir, f'cam{idx:03d}'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, f'cam{idx:03d}', 'frames'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, f'cam{idx:03d}', 'masks'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, f'cam{idx:03d}', 'times'), exist_ok=True)

            persp_frame = pers_imgs_l[p].permute(1, 2, 0).cpu().numpy()
            pers_depth = pers_depths_l[p].permute(1, 2, 0).cpu().numpy()[..., 0]
            #print(c2w_.shape, persp_frame.shape, pers_depth.shape)
            cv2.imwrite(os.path.join(output_dir, f'cam{idx:03d}', 'frames', f'{i:05d}.png'), persp_frame)
            cv2.imwrite(os.path.join(output_dir, f'cam{idx:03d}', 'masks', f'{i:05d}.png'), (np.ones_like(persp_frame[..., 0]))*255)
            np.save(os.path.join(output_dir, f'cam{idx:03d}', f'traj.npy'), c2w_[:3, :])
            with open(os.path.join(output_dir, f'cam{idx:03d}', 'times', f'{i:05d}.txt'), 'w') as f:
                f.write(str(i/(len_frames-1)))

    # 保存相机参数
    with open(os.path.join(output_dir, 'camera_params.txt'), 'w') as f:
        f.write(f'fov_x: {fov_x_s} {fov_x_l}\nfov_y: {fov_y_s} {fov_y_l}\n')
        f.write(f'focal_x: {camera_params_s.focal[0]} {camera_params_l.focal[0]}\nfocal_y: {camera_params_s.focal[1]} {camera_params_l.focal[1]}\n')
        f.write(f'num_cams: {pers_num*traj_num} {pers_num}\n')
        f.write(f'frame_count: {len_frames}\n')
        f.write(f'height: {height} {height}\n')
        f.write(f'width: {width} {width}\n')
    
    print('Data format for 4D Reconstruction prepared successfully in ', output_dir)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='inputs', help='Input directory of the scene')
    parser.add_argument('--video_type', type=str, default='enhance', choices=['guidance', 'refinement', 'enhancement'], help='Suffix of the video')
    parser.add_argument('--min_depth', type=int, default=2, help='Min depth of the scene')
    parser.add_argument('--max_depth', type=int, default=10, help='Max depth of the scene')

    parser.add_argument('--perturb', type=float, default=0.3, help='Perturbation of the camera')
    parser.add_argument('--tex_threshold', type=float, default=20.0, help='Threshold of texture change std')
    parser.add_argument('--pers_res', type=int, default=512, help='Resolution of the perspective images')
    parser.add_argument('--fov_ratio_s', type=float, default=1.0, help='Ratio of the first fov')
    parser.add_argument('--fov_ratio_l', type=float, default=1.3, help='Ratio of the second fov')
    
    args = parser.parse_args()
    input_dir = args.input_dir
    video_type = args.video_type
    scene_name = os.path.basename(input_dir)
    video_path = os.path.join(input_dir, f'{scene_name}_{video_type}.mp4')

    dirname = os.path.basename(input_dir)
    flow_dir = os.path.join(input_dir, 'flow') # directory of optical flow
    os.makedirs(flow_dir, exist_ok=True)
    depth_dir = os.path.join(input_dir, 'depth') # directory of depth
    os.makedirs(depth_dir, exist_ok=True)


    gpus_list = [0]
    cuda_devices = ["cuda:" + str(gpu) for gpu in gpus_list]
    device = random.choice(cuda_devices)
    args.device = device

    video_reader = VideoReader(video_path, ctx=cpu(0))
    video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()
    video = [frame for frame in video]
    height, width, _ = video[0].shape
    fps = video_reader.get_avg_fps()

    print('Frames length:', len(video))
    print('Resolution:', height, width)
    print('FPS:', fps)

    mask_pre_list, mask_post_list = abstract_flow(video, flow_dir, device, fps, args)
    print('#---------------Extracted flow masks successfully---------------#')
    geo_predictor = PanoGeoPredictor(frame_list=video, mask1=mask_pre_list, mask2=mask_post_list)
    distances = geo_predictor()
    print('#---------------Estimated depth successfully---------------#')

    distances = torch.stack(distances, dim=0)
    distances = scale_unit(distances.cpu().numpy())
    save_video((distances*255).astype(np.uint8), depth_dir, 'distance.mp4', fps=fps)  # save depth video

    ### rescale depth ###
    distances = distances * (args.max_depth - args.min_depth) + args.min_depth
    print(f'Depth map saved in {os.path.join(input_dir, "depth")}')

    data_dir = os.path.join(input_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    print('#---------------Generate training data for 4D Reconstruction---------------#')
    generate_temporal_point_cloud(video, distances, mask_pre_list, mask_post_list, data_dir, args)
    gen_train_data(video, distances, mask_pre_list, mask_post_list, data_dir, args)
    