# based on 360DVD
# https://github.com/Akaneqwq/360DVD/blob/main/scripts/video2flow.py


import argparse
from PanoFlowAPI.apis.PanoRaft import PanoRAFTAPI
from decord import VideoReader, cpu

import torch
import numpy as np
import cv2
import os

import random
import torch
from utils.inference_utils import *


def abstract_flow(frames_list, dir_path, device, fps, args=None):

    # Add args for PanoFlow
    if args is None:
        raise ValueError("args is None")
    args.dataset = None
    args.train = False
    args.eval_iters = 12

    ckpt_path = "./checkpoints/PanoFlow(RAFT)-wo-CFE.pth"
    if not os.path.exists(ckpt_path):
        print("Downloading PanoFlow checkpoint...")
        os.system("gdown https://drive.google.com/uc?id=103ToTG4xVYzn87FZ_s4fBT0rT93JnYnA -O ./checkpoints/")

        
    flow_estimater = PanoRAFTAPI(
        device=device, model_path=ckpt_path, args=args
    )

    os.makedirs(dir_path, exist_ok=True)
    output_path = os.path.join(dir_path, "opticalflow.mp4")
    last_flow_img = None
    last_frame = None
    flows_list = []
    flows_vis_list = []

    height, width = frames_list[0].shape[:2]

    for i in range(len(frames_list)):
        frame = frames_list[i]
        frame = cv2.resize(frame, (1024, 512))
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device).float()
        if last_frame is not None:
            flow = flow_estimater.estimate_flow_cfe(last_frame, frame)
            flows_list.append(flow.squeeze().cpu().numpy())
            flow_img = flow_estimater.flow2img(flow, alpha=0.1, max_flow=25)
            flow_img = flow_img[0].numpy()
            flow_img = 255 - flow_img
            flows_vis_list.append(flow_img)
        last_frame = frame

    flows_vis = np.array(flows_vis_list)
    save_video(flows_vis, dir_path, "opticalflow.mp4", fps=fps)

    # Make mask
    flows = np.array(flows_list)
    mask_pre_list = []
    mask_post_list = []

    for i in range(flows.shape[0]):
        flow = flows[i]

        mask = (np.abs(flow[..., 0]) > 1) | (np.abs(flow[..., 1]) > 1)
        mask = mask.astype(np.uint8)

        h, w = flow.shape[:2]
        flow_map = np.zeros_like(flow, dtype=np.float32)
        flow_map[..., 0] = (np.arange(w) + flow[..., 0]).astype(int) % w
        flow_map[..., 1] = (np.arange(h)[:, np.newaxis] + flow[..., 1]).astype(int) % h


        flow_index = np.where(mask.flatten() == 1)
        warped_mask = np.zeros_like(mask, dtype=np.uint8)
        index = flow_map.reshape(-1, 2)[flow_index].astype(int)
        warped_mask[index[:, 1], index[:, 0]] = 1

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.resize(mask, (width, height))
        mask = (mask > 0).astype(np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        warped_mask = cv2.resize(warped_mask, (width, height))
        warped_mask = (warped_mask > 0).astype(np.uint8)
        warped_mask = cv2.dilate(warped_mask, kernel, iterations=2)

        mask_pre_list.append(mask)
        mask_post_list.append(warped_mask)
    
    # save mask
    mask_pre_vis = np.array([np.repeat(mask[:, :, np.newaxis], 3, axis=2) * 255 for mask in mask_pre_list])
    mask_post_vis = np.array([np.repeat(mask[:, :, np.newaxis], 3, axis=2) * 255 for mask in mask_post_list])
    save_video(mask_pre_vis, dir_path, "mask_pre.mp4", fps=fps)
    save_video(mask_post_vis, dir_path, "mask_post.mp4", fps=fps)
    
    print(f'Flow mask saved in {dir_path}')

    return mask_pre_list, mask_post_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video path')
    parser.add_argument('-o', '--output', type=str, default='outputs', help='Output folder')
    args = parser.parse_args()
    args.dataset = None
    args.train = False
    args.eval_iters = 12

    gpus_list = [0]
    cuda_devices = ["cuda:" + str(gpu) for gpu in gpus_list]
    
    os.makedirs(args.output, exist_ok=True)

    video_path = args.input
    flow_path = args.output

    device = random.choice(cuda_devices)

    video_reader = VideoReader(video_path, ctx=cpu(0))
    video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()
    video = [frame for frame in video]
    height, width, _ = video[0].shape
    fps = video_reader.get_avg_fps()

    print('Frames length:', len(video))
    print('Resolution:', height, width)
    print('FPS:', fps)

    abstract_flow(video, flow_path, device, fps=10, args=args)

