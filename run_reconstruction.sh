device=0

# Directory of the scene
data_dir=data
scene_dir=$data_dir/scene_0


# Estimate Depth and prepare training data
CUDA_VISIBLE_DEVICES=$device python3 scripts/gen_train_data.py \
--input_dir $scene_dir \
--video_type enhancement \

# Train the spacetime gaussian model (lite or full)
CUDA_VISIBLE_DEVICES=$device python scripts/train_gaussian.py \
--config configs/lite_spacetime/pano_train.json \
--model_path $scene_dir/3d \
--source_path $scene_dir/data \
--loader panorama

# Render the spacetime gaussian model using train data
CUDA_VISIBLE_DEVICES=$device python scripts/render_gaussian.py \
--skip_train \
--configpath configs/lite_spacetime/pano_train.json \
--model_path $scene_dir/3d \
--source_path $scene_dir/data \
--loader panorama \
--save_folder train_data \
--has_gt
