device=0
seed=1024

ckpt_guide=checkpoints/holotime_guidance.ckpt
ckpt_refine=checkpoints/holotime_refinement.ckpt
config_guide=configs/inference_512_v1.0.yaml
config_refine=configs/inference_1024_v1.0_edit.yaml

# Directory
prompt_dir=input
save_dir=data

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

# Refinement Model
CUDA_VISIBLE_DEVICES=$device python3 scripts/inference_animator.py \
--seed ${seed} \
--ckpt_path $ckpt_refine \
--config $config_refine \
--savedir $save_dir \
--save_suffix refinement \
--n_samples 1 \
--bs 1 --height 576 --width 1024 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--guidance_dir $save_dir \
--guidance_suffix guidance \
--text_input \
--video_length 25 \
--frame_stride 10 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae

# Optional: Enhance the video
# Super resolution and frame interpolation
CUDA_VISIBLE_DEVICES=$device python3 scripts/enhance_video.py \
--version v2 \
--up_scale 2 --target_fps 20 --noise_aug 100 \
--solver_mode 'fast' --steps 15 \
--input_path $save_dir \
--prompt_path $prompt_dir/text_prompts.txt \
--save_dir $save_dir \
--suffix enhancement \
--sr_x2
