```bash
https://github.com/MattWallingford/360-1M
https://github.com/darijo/360-Video-DASH-Dataset
http://www.svcl.ucsd.edu/projects/AVSpatialAlignment/#download

https://huggingface.co/ysmikey/Layerpano3D-FLUX-Panorama-LoRA

https://github.com/ProGamerGov/ComfyUI_preview360panorama
https://github.com/ProGamerGov/ComfyUI_preview360panorama/issues/5

https://github.com/jpgallegoar/ComfyUI_preview360panorama

https://github.com/PKU-YuanGroup/HoloTime



sudo apt-get update && sudo apt-get install cbm ffmpeg git-lfs

git clone https://github.com/PKU-YuanGroup/HoloTime --recursive
cd HoloTime
conda create -n holotime python=3.10 -y
conda activate holotime
pip install ipykernel
python -m ipykernel install --user --name holotime --display-name "holotime"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 
pip install -r requirements.txt

vim /environment/miniconda3/envs/holotime/lib/python3.10/site-packages/basicsr/data/degradations.py
from torchvision.transforms.functional_tensor import rgb_to_grayscale
to
from torchvision.transforms._functional_tensor import rgb_to_grayscale


device=0
seed=1024

ckpt_guide=checkpoints/holotime_guidance.ckpt
ckpt_refine=checkpoints/holotime_refinement.ckpt
config_guide=configs/inference_512_v1.0.yaml
config_refine=configs/inference_1024_v1.0_edit.yaml

# Directory
prompt_dir=input_test
save_dir=data_test

# Guidance Model
CUDA_VISIBLE_DEVICES=$device python3 scripts/inference_animator.py \
--seed ${seed} \
--ckpt_path $ckpt_guide \
--config $config_guide \
--savedir $save_dir \
--save_suffix guidance \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 25 \
--frame_stride 10 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae
```

```python
from scripts.inference_animator import *
import datetime
import os

# Set parameters
device = 0
seed = 1024

ckpt_guide = "checkpoints/holotime_guidance.ckpt"
ckpt_refine = "checkpoints/holotime_refinement.ckpt"
config_guide = "configs/inference_512_v1.0.yaml"
config_refine = "configs/inference_1024_v1.0_edit.yaml"

# Directory paths
prompt_dir = "input_test"
save_dir = "data_test"

# Create a namespace object to hold all arguments
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Guidance Model parameters
args = Namespace(
    seed=seed,
    ckpt_path=ckpt_guide,
    config=config_guide,
    savedir=save_dir,
    save_suffix="guidance",
    n_samples=1,
    bs=1,
    height=320,
    width=512,
    unconditional_guidance_scale=7.5,
    ddim_steps=50,
    ddim_eta=1.0,
    prompt_dir=prompt_dir,
    text_input=True,
    video_length=25,
    frame_stride=10,
    timestep_spacing="uniform_trailing",
    guidance_rescale=0.7,
    perframe_ae=True,
    # These are not in the command but needed by the function
    guidance_dir=None,
    guidance_suffix="guide",
    negative_prompt=False,
    multiple_cond_cfg=False,
    cfg_img=None,
    loop=False,
    interp=False
)

# Make sure directories exist
os.makedirs(save_dir, exist_ok=True)

# Set up CUDA device
import torch
torch.cuda.set_device(device)

# Run the inference
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print("#-------------------------Panoramic Animator Inference: %s -------------------------#" % now)

from pytorch_lightning import seed_everything
seed_everything(seed)

# Run the function directly
#run_inference(args, gpu_num=1, gpu_no=0)

def run_inference_samples(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae

    if not os.path.exists(args.ckpt_path):
        print(f"Downloading the Panoramic Animator checkpoint...")
        hf_hub_download(repo_id='Marblueocean/HoloTime', filename=f'holotime_{args.save_suffix}.ckpt', local_dir=os.path.dirname(args.ckpt_path))
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    # os.makedirs(fakedir, exist_ok=True)
    os.makedirs(args.savedir, exist_ok=True)

    ## prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    filename_list, data_list, prompt_list, video_list = load_data_prompts(args.prompt_dir, args.guidance_dir, args.guidance_suffix, video_size=(args.height, args.width), video_frames=n_frames, interp=args.interp, device_num=gpu_no)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]
    if args.guidance_dir:
        video_list_rank = [video_list[i] for i in indices]

    start = time.time()
    req = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice+args.bs]
            filenames = filename_list_rank[indice:indice+args.bs]
            images = data_list_rank[indice:indice+args.bs]
            if isinstance(images, list):
                images = torch.stack(images, dim=0).to("cuda")
            else:
                images = images.unsqueeze(0).to("cuda")
            
            if args.guidance_dir:
                video_inputs = video_list_rank[indice:indice+args.bs]
                if isinstance(video_inputs, list):
                    video_inputs = torch.stack(video_inputs, dim=0).to("cuda")
                else:
                    video_inputs = video_inputs.unsqueeze(0).to("cuda")
            else:
                video_inputs = None

            batch_samples = image_guided_synthesis(model, prompts, images, video_inputs, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.loop, \
                                args.interp, args.timestep_spacing, args.guidance_rescale)
            req.append(batch_samples)
    return req

import torch
import cv2
import numpy as np

def save_tensor_as_video_cv2(tensor, output_path, fps=30):
    # 预处理张量
    tensor = tensor.squeeze().permute(1, 2, 3, 0)  # [T, H, W, C]
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    frames = tensor.cpu().numpy()

    # 配置 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (frames.shape[2], frames.shape[1])  # (宽度, 高度)
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # 写入帧
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

save_tensor_as_video_cv2(samples[0], "output.mp4")
```
