export CKPT="your_path_to_checkpoint"
export WORK_DIR="your_path_to_work_dir"
export DATA_DIR="your_path_to_datasets"
export CLIP_TEXT_CKPT="path_to_clip_text_encoder_checkpoint"
export CLIP_IMAGE_CKPT="path_to_clip_image_encoder_checkpoint"
export CONFIG_FILE="path_to_sam2_config_file"
export SAM2_CKPT="path_to_sam2_checkpoint"
# export HF_TOKEN=xxxxx
# export MASTER_ADDR=xxx.xxx.xxx.xxx
# export MASTER_PORT=xxxx

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
--resume $SEGVOL_CKPT \
--work_dir $WORK_DIR \
--data_dir $DATA_DIR \
--clip_text_ckpt $CLIP_TEXT_CKPT \
--clip_image_ckpt $CLIP_IMAGE_CKPT \
--config_file $CONFIG_FILE\
--sam2_ckpt SAM2_CKPT