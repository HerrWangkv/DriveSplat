#!/bin/bash

mkdir -p /data/raw
mkdir -p /data/cache
fusermount -u /data/raw || true
fusermount -u /data/cache || true
squashfuse /data/nuscenes.sqfs /data/raw/
squashfuse /data/cache.sqfs /data/cache

export MODEL_NAME="jingheya/lotus-depth-g-v2-1-disparity"

# training dataset
export DATASET_CONFIG_PATH="data/NuScenes/sdaig.yaml"

# training configs
export BATCH_SIZE=1
export CUDA=01234567
export GAS=1
export TOTAL_BSZ=$(($BATCH_SIZE * ${#CUDA} * $GAS))

# model configs for multi-step training
export MAX_TIMESTEPS=1000
export VAL_STEP=500
export NUM_INFERENCE_STEPS=20  # Number of denoising steps during validation

# output dir
export OUTPUT_DIR="output/train-sdaig-pred-multistep-bsz${TOTAL_BSZ}/"

accelerate launch --config_file=accelerate_configs/$CUDA.yaml --mixed_precision="fp16" \
  train_sdaig_pred.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_config_path=$DATASET_CONFIG_PATH \
  --dataloader_num_workers=0 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed=42 \
  --max_train_steps=20000 \
  --learning_rate=3e-05 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --multi_step_training \
  --max_timesteps=$MAX_TIMESTEPS \
  --num_inference_steps=$NUM_INFERENCE_STEPS \
  --validation_steps=$VAL_STEP \
  --checkpointing_steps=$VAL_STEP \
  --output_dir=$OUTPUT_DIR \
  --checkpoints_total_limit=2 \
  --resume_from_checkpoint="latest"
