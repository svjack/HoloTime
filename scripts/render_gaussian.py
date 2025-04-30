# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ========================================================================================================
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the thirdparty/gaussian_splatting/LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys 
sys.path.append("./thirdparty/gaussian_splatting")
sys.path.insert(1, os.path.join(sys.path[0], '..', ))
import torch
from tqdm import tqdm
from os import makedirs
import torchvision
import time 
import scipy
import numpy as np 
import warnings
import json 
import matplotlib.pyplot as plt

from thirdparty.gaussian_splatting.scene import Scene
from helper_train import getrenderpip, getmodel, trbfunction
from thirdparty.gaussian_splatting.helper3dg import gettestparse
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams

from utils.inference_utils import save_video
warnings.filterwarnings("ignore")

# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/render.py and https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, rbfbasefunction, rdpip, has_gt, make_video):
    render, GRsetting, GRzer = getrenderpip(rdpip) 
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if gaussians.rgbdecoder is not None:
        gaussians.rgbdecoder.cuda()
        gaussians.rgbdecoder.eval()

    scene_dir = model_path
    image_names = []

    if rdpip == "train_ours_full":
        # full model faster now when use fuse the rendering part (MLP) into cuda, same as 3dgs and instant-NGP. 
        render, GRsetting, GRzer = getrenderpip("test_ours_full_fused")
    elif rdpip == "train_ours_lite":
        render, GRsetting, GRzer = getrenderpip("test_ours_lite") 
    elif rdpip == "train_ours_fullss":
        render, GRsetting, GRzer = getrenderpip("test_ours_fullss_fused") # 
    elif rdpip == "train_ours_litess":
        render, GRsetting, GRzer = getrenderpip("test_ours_litess") # 
    
    else:
        render, GRsetting, GRzer = getrenderpip(rdpip) 

    render_list = []
    depth_list = []
    gt_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering and metric progress")):
        renderingpkg = render(view, gaussians, pipeline, background, scaling_modifier=1.0, basicfunction=rbfbasefunction,  GRsetting=GRsetting, GRzer=GRzer) # C x H x W
        rendering = renderingpkg["render"]
        rendering = torch.clamp(rendering, 0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rendernumpy = rendering.permute(1,2,0).detach().cpu().numpy()
        render_list.append(((rendernumpy)*255).astype(np.uint8)) 
        
        if has_gt:
            gt = view.original_image[0:3, :, :].cuda().float()
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_numpy = gt.permute(1,2,0).detach().cpu().numpy()
            gt_list.append(((gt_numpy)*255).astype(np.uint8))

    if make_video:
        save_video(render_list, render_path, "video.mp4", fps=30)




def run_test(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, rgbfunction="rgbv1", rdpip="v2", loader="colmap", save_folder='test', has_gt=False, make_video=True):
    
    with torch.no_grad():
        print("use model {}".format(dataset.model))
        GaussianModel = getmodel(dataset.model) # default, gmodel, we are tewsting 

        gaussians = GaussianModel(dataset.sh_degree, rgbfunction)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, loader=loader)
        rbfbasefunction = trbfunction
        numchannel = 9
        #bg_color =  [0 for _ in range(numchannels)]
        bg_color =  [1 for i in range(numchannel)] if dataset.white_background else [0 for i in range(numchannel)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if gaussians.ts is None :
            cameraslit = scene.getTrainCameras()
            H,W = cameraslit[0].image_height, cameraslit[0].image_width
            gaussians.ts = torch.ones(1,1,H,W).cuda()
                    
        render_set(dataset.model_path, save_folder, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, rbfbasefunction, rdpip, has_gt, make_video)


if __name__ == "__main__":
    

    args, model_extract, pp_extract, multiview =gettestparse()
    run_test(model_extract, args.test_iteration, pp_extract, args.skip_train, args.skip_test, rgbfunction=args.rgbfunction, rdpip=args.rdpip, loader=args.loader, save_folder=args.save_folder, has_gt=args.has_gt, make_video=args.make_video)
