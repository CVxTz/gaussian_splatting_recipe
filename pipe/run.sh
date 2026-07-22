TRIAL="trial11"
TRIAL_PATH="/media/${USER}/0138F8B62CE26963/colmap/${TRIAL}"
VIDEO=/media/${USER}/0138F8B62CE26963/videos/VID_20260722_163524.mp4
python gaussian_splatting/pipeline_hloc.py \
    --video ${VIDEO} \
    --output_dir ${TRIAL_PATH}/ \
    --fps 2 \
    --estimate_distortion \
    --retrieval_model megaloc

python gaussian_splatting/visualize_matches.py \
    --image_dir ${TRIAL_PATH}/images \
    --features ${TRIAL_PATH}/hloc_outputs/features_disk.h5 \
    --matches ${TRIAL_PATH}/hloc_outputs/matches_disk.h5 \
    --output_dir ${TRIAL_PATH}/hloc_outputs/visualizations \
    --num_pairs 10

python gaussian_splatting/train_splat.py \
    --trial_dir ${TRIAL_PATH}/ \
    --images_path ${TRIAL_PATH}/images \
    --masks_path ${TRIAL_PATH}/masks \
    --output_splat ${TRIAL_PATH}/gaussian_model.ply \
    --epochs 100

python gaussian_splatting/render_interpolated_video.py \
    --trial_dir ${TRIAL_PATH}/ \
    --ply_path ${TRIAL_PATH}/gaussian_model.ply \
    --output_video ${TRIAL_PATH}/rendered_flythrough.mp4 \
    --fps 30 \
    --seconds 50
