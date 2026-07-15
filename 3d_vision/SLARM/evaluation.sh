#!/bin/bash

# NPU platform
export DEVICE_TYPE="NPU"
export ASCEND_RT_VISIBLE_DEVICES=0  # NPU device id to run on
# NPU performance optimization
export AVOID_AI_CPU=1  # Avoid generating AI_CPU Sin and Cos operators due to double data type.
export USE_EQUAL_CROSS=1  # Equivalent replacement for torch.cross
export TASK_QUEUE_ENABLE=2  # Speed up host distribution
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True  # Start virtual memory to save device memory
export CONTEXT_FEAT=1  # No feature is rendered; feature is only supervised from the input viewpoint.

# Set model / dataset configuration
export FEAT_DIST=1
export DATASET=waymo
export DATA_ROOT= # path to the dataset
export OVERFIT_EXP=1

export MASTER_PORT=16844
export DEVICE_NUM=1
export PROJECT=slarm
export EXP_NAME=eval_0624
export CKPT_PTH= # path to the checkpoint of SLARM

torchrun --nproc_per_node=$DEVICE_NUM --master_port ${MASTER_PORT} eval_slarm.py \
    --evaluate \
    --project ${PROJECT} \
    --exp_name ${EXP_NAME} \
    --dataset ${DATASET} \
    --data_root $DATA_ROOT \
    --eval_batch_size 1 \
    --model slarm \
    --load_depth --load_flow --load_ground \
    --load_semantic_label \
    --num_max_cameras 3 --use_affine_token \
    --sigmoid_rgb \
    --num_motion_tokens 0 \
    --use_sky_token \
    --embed_dim 768 --depth 12 --patch_embed conv --patch_size 8 \
    --use_ms3_motion \
    --use_last_token \
    --shortcut_rgb \
    --add_patch_plucker_embed \
    --similarity_probs_threshold 0.2 \
    --enable_tensorboard \
    --load_from $CKPT_PTH
