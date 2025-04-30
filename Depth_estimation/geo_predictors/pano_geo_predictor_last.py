# based on DreamScene360
# https://github.com/ShijieZhou-UCLA/DreamScene360/tree/main/geo_predictors


import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers
from tqdm import tqdm
from PIL import Image

import numpy as np
import cv2 as cv
import tinycudann as tcnn

from .geo_predictor import GeoPredictor
from Depth_estimation.fields.networks import VanillaMLP
from Depth_estimation.utils.geo_utils import panorama_to_pers_directions
from Depth_estimation.utils.camera_utils import *

def scale_unit(x):
    return (x - x.min()) / (x.max() - x.min())


class GeometricField(nn.Module):
    def __init__(self,
                n_levels=16,
                log2_hashmap_size=19,
                base_res=16,
                fine_res=2048):
        super().__init__()
        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1))
        self.hash_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            }
        )

        self.geo_mlp = VanillaMLP(dim_in=n_levels * 2 + 3,
                                  dim_out=1,
                                  n_neurons=64,
                                  n_hidden_layers=2,
                                  sphere_init=True,
                                  weight_norm=False)

    def forward(self, directions, requires_grad=False):
        if requires_grad:
            if not self.training:
                directions = directions.clone()  # get a copy to enable grad
            directions.requires_grad_(True)

        dir_scaled = directions * 0.49 + 0.49
        selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32)
        scene_feat = self.hash_grid(dir_scaled)

        distance = F.softplus(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0] + 1.)

        if requires_grad:
            grad = torch.autograd.grad(
                distance, directions, grad_outputs=torch.ones_like(distance),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return distance, grad
        else:
            return distance


class PanoGeoPredictor(GeoPredictor):
    def __init__(self, frame_list, mask1, mask2):
        super().__init__()
        self.frame_list = frame_list
        self.mask1 = mask1
        self.mask2 = mask2
        self.depth_predictor = diffusers.MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16).to("cuda")

    def grads_to_normal(self, grads):
        grads = grads.cpu()
        height, width, _ = grads.shape
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        ortho_a = torch.randn([height, width, 3])
        ortho_b = torch.linalg.cross(pano_dirs, ortho_a)
        ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
        ortho_a = torch.linalg.cross(ortho_b, pano_dirs)
        ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

        val_a = (grads * ortho_a).sum(-1, True) * pano_dirs + ortho_a
        val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
        val_b = (grads * ortho_b).sum(-1, True) * pano_dirs + ortho_b
        val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)

        normals = torch.cross(val_a, val_b)
        normals = normals / torch.linalg.norm(normals, 2, -1, True)
        is_inside = ((normals * pano_dirs).sum(-1, True) < 0.).float()
        normals = normals * is_inside + -normals * (1. - is_inside)
        return normals.cuda()

    def __call__(self, gen_res=512, reg_loss_weight=1e-1,):

        length = len(self.frame_list)
        frames = np.array(self.frame_list) / 255.
        img = torch.tensor(frames[0]).float().cuda().permute(2, 0, 1)
        height, width, = img.shape[1], img.shape[2]
        device = img.device

        ####################################### Prepare data and parameters #######################################
        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []
        ratio = 1.1
        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=ratio)

        fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
        fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
        cx = torch.ones_like(fx) * gen_res * .5
        cy = torch.ones_like(fy) * gen_res * .5

        pers_dirs = pers_dirs.to(device)
        pers_ratios = pers_ratios.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                              dim=1)

        rot_c2w = torch.linalg.inv(rot_w2c)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        pers_imgs = F.grid_sample(img[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]

        ####################################### Space Optimization for first frame #####################################################

        pers_pred_depth = []  # Perspective depth maps from pretrained model
        image_pils = [Image.fromarray((pers.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)) for pers in pers_imgs]
        pred_depths = self.depth_predictor(image_pils).prediction

        for i in range(n_pers):
            with torch.no_grad():
                pred_depth = pred_depths[i: i+1]
                pred_depth = torch.tensor(np.array(pred_depth)).float().cuda().clip(0., None).permute(0, 3, 1, 2)  # [1, 1, res, res]
                pred_depth = pred_depth / (pred_depth.mean() + 1e-5)
                
                pred_depth_ratio = pred_depth * pers_ratios[i].permute(2, 0, 1)[None]
                pers_pred_depth.append(pred_depth_ratio)

        pers_pred_depth = torch.cat(pers_pred_depth, dim=0)  # [n_pers, 1, res, res]
        pers_dirs = pers_dirs.permute(0, 3, 1, 2)

        sup_infos = torch.cat([pers_dirs, pers_pred_depth], dim=1)

        scale_params = torch.zeros([n_pers], requires_grad=True, device=device)
        bias_params_local_distance  = torch.zeros([n_pers, 1, gen_res, gen_res], requires_grad=True, device=device)
        geo_field = GeometricField(fine_res = width).to(device)

        # Hyperparameters
        all_iter_steps = 1500
        lr_alpha = 1e-2
        init_lr = 1e-1
        init_lr_sp = 1e-2
        init_lr_local = 1e-1
        local_batch_size = 256

        optimizer_sp = torch.optim.Adam(geo_field.parameters(), lr=init_lr_sp)
        optimizer_global = torch.optim.Adam([scale_params], lr=init_lr)
        optimizer_local = torch.optim.Adam([bias_params_local_distance], lr=init_lr_local)
        
        ########### Two phases optimization ##############
        for phase in ['global', 'hybrid']:
            ema_loss_for_log = 0.0
            progress_bar = tqdm(range(1, all_iter_steps + 1), desc="Frame 0")
            loss_vis = []
            
            for iter_step in range(1, all_iter_steps + 1):
                progress = iter_step / all_iter_steps
                if phase == 'global':
                    progress = progress * .5
                else:
                    progress = progress * .5 + .5

                lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
                for g in optimizer_global.param_groups:
                    g['lr'] = init_lr * lr_ratio
                for g in optimizer_local.param_groups:
                    g['lr'] = init_lr_local * lr_ratio
                for g in optimizer_sp.param_groups:
                    g['lr'] = init_lr_sp * lr_ratio

                sample_coords = torch.rand(n_pers, local_batch_size, 1, 2, device=device) * 2. - 1           # [n_pers, local_batch_size, 1, 2] range (-1, +1)
                cur_sup_info = F.grid_sample(sup_infos, sample_coords, padding_mode='border')        # [n_pers, 4, local_batch_size, 1]
                distance_bias = F.grid_sample(bias_params_local_distance, sample_coords, padding_mode='border')  # [n_pers, 1, local_batch_size, 1]
                distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)                              # [n_pers, local_batch_size, 1]

                dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)

                ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances * F.softplus(scale_params[:, None, None])  # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances + distance_bias

                pred_distances = geo_field(dirs.reshape(-1, 3), requires_grad=False)
                pred_distances = pred_distances.reshape(n_pers, local_batch_size, 1)
                distance_loss = F.smooth_l1_loss(ref_pred_distances, pred_distances, beta=5e-1, reduction='mean')
                reg_loss = ((F.softplus(scale_params).mean() - 1.)**2).mean()

                if phase == 'hybrid':
                    distance_bias_local = bias_params_local_distance
                    distance_bias_tv_loss = F.smooth_l1_loss(distance_bias_local[:, :, 1:, :], distance_bias_local[:, :, :-1, :], beta=1e-2) + \
                                            F.smooth_l1_loss(distance_bias_local[:, :, :, 1:], distance_bias_local[:, :, :, :-1], beta=1e-2)

                else:
                    distance_bias_tv_loss = 0.
                loss = distance_loss +\
                       reg_loss * reg_loss_weight +\
                       distance_bias_tv_loss 
              
                optimizer_global.zero_grad()
                optimizer_sp.zero_grad()
                if phase == 'hybrid':
                    optimizer_local.zero_grad()

                loss.backward()
                optimizer_global.step()
                optimizer_sp.step()
                if phase == 'hybrid':
                    optimizer_local.step()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    loss_vis.append(ema_loss_for_log) 
                    if iter_step % 1 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                        progress_bar.update(1)
                    if iter_step == all_iter_steps:
                        progress_bar.close()
        
        depth_video = []
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width)).cuda()
        new_distances = geo_field(pano_dirs.reshape(-1, 3), requires_grad=False)
        new_distances = new_distances.detach().reshape(height, width, 1)
        depth_video.append(new_distances)


        ####################################### Space-Time Optimization for subsequent frames #####################################################

        flow_masks1 = np.array(self.mask1)
        flow_masks2 = np.array(self.mask2)
        flow_masks = flow_masks1 | flow_masks2

        first_depth = new_distances.permute(2, 0, 1).cuda()

        union_flow_mask = np.max(flow_masks, axis=0)
        union_flow_mask = scale_unit(torch.tensor(union_flow_mask).float().cuda()).unsqueeze(0)

        sample_coords = img_coord_to_sample_coord(img_coords)
        pers_union_mask = F.grid_sample(union_flow_mask[None].expand(n_pers, -1, -1, -1), sample_coords[:20], padding_mode='border') # [n_pers, 1, gen_res, gen_res]
        pers_first_refdepth = F.grid_sample(first_depth[None].expand(n_pers, -1, -1, -1), sample_coords[:20], padding_mode='border') # [n_pers, 1, gen_res, gen_res]

        for frame_index in range(length-1): 

            prev_depth = new_distances.permute(2, 0, 1).cuda()
            pers_prev_refdepth = torch.zeros_like(pers_pred_depth).cuda()

            flow_mask = flow_masks[frame_index]
            if frame_index < length-2:
                flow_mask1 = flow_masks1[frame_index+1]
                flow_mask = flow_mask | flow_mask1
            flow_mask = scale_unit(torch.tensor(flow_mask).float().cuda()).unsqueeze(0)

            #### Choose the perspective views to be optimized ####
            index = np.zeros(n_pers)    
            for j in range(n_pers):
                ico_mask = torch.zeros((img.shape[1], img.shape[2]), dtype=torch.uint8).cuda()
                x_coords = (img_coords[j][..., 0].reshape(-1)*height).int().clip(0, height-1)
                y_coords = (img_coords[j][..., 1].reshape(-1)*width).int().clip(0, width-1)
                ico_mask[x_coords, y_coords] = 1
                if (flow_mask.int() & ico_mask.int()).any():
                    index[j] = 1
                
            if index.any():
                frame = torch.tensor(frames[frame_index+1]).float().cuda().permute(2, 0, 1)
                sample_coords = img_coord_to_sample_coord(img_coords)
                pers_imgs = F.grid_sample(frame[None].expand(n_pers, -1, -1, -1), sample_coords[:20], padding_mode='border') # [n_pers, 3, gen_res, gen_res]
                pers_cur_mask = F.grid_sample(flow_mask[None].expand(n_pers, -1, -1, -1), sample_coords[:20], padding_mode='border') # [n_pers, 1, gen_res, gen_res]
                pers_prev_refdepth = F.grid_sample(prev_depth[None].expand(n_pers, -1, -1, -1), sample_coords[:20], padding_mode='border') # [n_pers, 1, gen_res, gen_res]

                image_pils = [Image.fromarray((pers.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)) for pers in pers_imgs]
                pred_depths = self.depth_predictor(image_pils).prediction
                for i in range(n_pers):
                    if index[i] == 1:
                        with torch.no_grad():
                            pred_depth = pred_depths[i: i+1]
                            pred_depth = torch.tensor(np.array(pred_depth)).float().cuda().clip(0., None).permute(0, 3, 1, 2)  # [1, 1, res, res]
                            pred_depth = pred_depth / (pred_depth.mean() + 1e-5)
                            pred_depth_ratio = pred_depth * pers_ratios[i].permute(2, 0, 1)[None]
                            pers_pred_depth[i]=pred_depth_ratio


                sup_infos = torch.cat([pers_dirs, pers_pred_depth, pers_prev_refdepth, pers_cur_mask, pers_first_refdepth, pers_union_mask], dim=1)
                all_iter_steps = 1000
                lr_alpha = 1e-2
                init_lr = 1e-1
                init_lr_sp = 1e-2
                init_lr_local = 1e-1
                local_batch_size = 256
                
                #### Beginning with the geo_field from previous frame ####
                scale_params = torch.zeros([n_pers], requires_grad=True, device=device)
                bias_params_local_distance  = torch.zeros([n_pers, 1, gen_res, gen_res], requires_grad=True, device=device)
                #geo_field = GeometricField(fine_res = 2048).cuda()
                optimizer_sp = torch.optim.Adam(geo_field.parameters(), lr=init_lr_sp)
                optimizer_global = torch.optim.Adam([scale_params], lr=init_lr)
                optimizer_local = torch.optim.Adam([bias_params_local_distance], lr=init_lr_local)
                
                index = torch.tensor(index).float()
                indices = torch.where(index == 1)[0]
                ###### Start optimization ######
                for phase in ['hybrid']:
                    ema_loss_for_log = 0.0
                    progress_bar = tqdm(range(1, all_iter_steps + 1), desc=f"Frame {frame_index+1}")
                    loss_vis = []

                    for iter_step in range(1, all_iter_steps + 1):
                        progress = iter_step / all_iter_steps
                        if phase == 'global':
                            progress = progress * .5
                        else:
                            progress = progress * .5 + .5

                        lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
                        for g in optimizer_global.param_groups:
                            g['lr'] = init_lr * lr_ratio
                        for g in optimizer_local.param_groups:
                            g['lr'] = init_lr_local * lr_ratio
                        for g in optimizer_sp.param_groups:
                            g['lr'] = init_lr_sp * lr_ratio

                        sample_coords = torch.rand(n_pers, local_batch_size, 1, 2, device=device) * 2. - 1           # [n_pers, local_batch_size, 1, 2] range (-1, +1)
                        cur_sup_info = F.grid_sample(sup_infos, sample_coords, padding_mode='border')        # [n_pers, 8, local_batch_size, 1]
                        distance_bias = F.grid_sample(bias_params_local_distance, sample_coords, padding_mode='border')  # [n_pers, 1, local_batch_size, 1]
                        distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)                             # [n_pers, local_batch_size, 1]

                        dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                        dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)

                        ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                        ref_pred_distances = ref_pred_distances * F.softplus(scale_params[:, None, None])   # [n_pers, local_batch_size, 1]
                        ref_pred_distances = ref_pred_distances + distance_bias

                        cur_mask = cur_sup_info[:, 5: 6, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                        prev_reference_distances = cur_sup_info[:, 4: 5, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                        
                        first_reference_distances = cur_sup_info[:, 6: 7, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                        first_mask = cur_sup_info[:, 7: 8, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                        #ref_anywhere_distance = cur_sup_info[:, 4: 5, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]

                        pred_distances = geo_field(dirs.reshape(-1, 3), requires_grad=False)
                        pred_distances = pred_distances.reshape(n_pers, local_batch_size, 1)
                        distance_loss = F.smooth_l1_loss(ref_pred_distances[indices], pred_distances[indices], beta=5e-1, reduction='mean')
                        reference_loss = F.smooth_l1_loss(prev_reference_distances*first_mask*(1-cur_mask), pred_distances*first_mask*(1-cur_mask), beta=5e-1, reduction='mean')
                        first_reference_loss = F.smooth_l1_loss(first_reference_distances*(1-first_mask), pred_distances*(1-first_mask), beta=5e-1, reduction='mean')
                        
                        reg_loss = ((F.softplus(scale_params[index.int()]).mean() - 1.)**2).mean()

                        if phase == 'hybrid':
                            distance_bias_local = bias_params_local_distance[indices]
                            distance_bias_tv_loss = F.smooth_l1_loss(distance_bias_local[:, :, 1:, :], distance_bias_local[:, :, :-1, :], beta=1e-2) + \
                                                    F.smooth_l1_loss(distance_bias_local[:, :, :, 1:], distance_bias_local[:, :, :, :-1], beta=1e-2)
                        else:
                            distance_bias_tv_loss = 0.
                        loss = distance_loss + reference_loss + first_reference_loss +\
                            reg_loss * reg_loss_weight +\
                            distance_bias_tv_loss 

                        optimizer_global.zero_grad()
                        optimizer_sp.zero_grad()
                        if phase == 'hybrid':
                            optimizer_local.zero_grad()

                        loss.backward()
                        optimizer_global.step()
                        optimizer_sp.step()
                        if phase == 'hybrid':
                            optimizer_local.step()

                        with torch.no_grad():
                            # Progress bar
                            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                            loss_vis.append(ema_loss_for_log) ###
                            if iter_step % 1 == 0:
                                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                                progress_bar.update(1)
                            if iter_step == all_iter_steps:
                                progress_bar.close()
                new_distances = geo_field(pano_dirs.reshape(-1, 3), requires_grad=False)
                new_distances = new_distances.detach().reshape(height, width, 1)
                depth_video.append(new_distances)
            else:
                depth_video.append(new_distances)

        return depth_video