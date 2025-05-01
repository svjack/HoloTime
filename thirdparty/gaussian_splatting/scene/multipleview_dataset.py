import os
import numpy as np
import math
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
#from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_folder,
        frame_count,
        focal1,
        focal2,
    ):
        self.focal1 = focal1
        self.focal2 = focal2
        self.frame_count = frame_count
        self.transform = T.ToTensor()

        self.image_paths, self.mask_paths, self.image_poses, self.image_times, self.image_names= self.load_images_path(cam_folder, cam_extrinsics)
        
    
    def load_images_path(self, cam_folder, cam_extrinsics):
        image_paths=[]
        mask_paths=[]
        image_poses=[]
        image_times=[]
        image_names=[]
        for idx, key in enumerate(cam_extrinsics):
            R = cam_extrinsics[idx][:3,:3]
            T = cam_extrinsics[idx][:3,3:4].reshape(-1)
            images_folder=os.path.join(cam_folder,"cam"+str(idx).zfill(3))

            image_range=range(self.frame_count)
            for i in image_range:    
                image_path=os.path.join(images_folder,"frames", str(i).zfill(5)+".png")
                image_paths.append(image_path)
                mask_path=os.path.join(images_folder,"masks", str(i).zfill(5)+".png")
                mask_paths.append(mask_path)
                image_poses.append((R,T))

                # open txt
                with open(os.path.join(images_folder,"times", str(i).zfill(5)+".txt"), "r") as f:
                    time = f.readline().strip()
                image_times.append(float(time))
                image_names.append(f"{idx}_{i}")
            
        return image_paths, mask_paths, image_poses, image_times, image_names
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]