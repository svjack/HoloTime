device=0

# Directory of the scene
data_dir=data
scene_dir=$data_dir/scene_0

# Generate the rendered trajectory [--trajectory] 
# choices: ['rotate360', 'rotateshow', 'back_and_forth', 'headbanging', 'llff']
# Applicable to all scenes
trajectory=headbanging
CUDA_VISIBLE_DEVICES=$device python scripts/gen_render_data.py \
--trajectory $trajectory \
--save_dir $data_dir

# Render the spacetime gaussian model using generated trajectory
CUDA_VISIBLE_DEVICES=$device python scripts/render_gaussian.py \
--skip_train \
--configpath configs/lite_spacetime/pano_train.json \
--model_path $scene_dir/3d \
--source_path $data_dir/$trajectory \
--loader panorama \
--save_folder $trajectory \
--make_video
