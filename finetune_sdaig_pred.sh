#!/bin/bash

mkdir -p /data/raw
mkdir -p /data/cache
squashfuse /data/nuscenes.sqfs /data/raw/
squashfuse /data/cache.sqfs /data/cache

export MODEL_NAME="output/box_depth&render_w_moving_objects/final"

# training dataset
export DATASET_CONFIG="data/NuScenes/sdaig.yaml"

# LoRA Configuration
export LORA_RANK=64
export LORA_ALPHA=64
export LORA_DROPOUT=0.1

# Training Configuration
export BATCH_SIZE=1
export CUDA=0
export GAS=1
export TOTAL_BSZ=$(($BATCH_SIZE * ${#CUDA} * $GAS))

# model configs for multi-step training
export MAX_TIMESTEPS=1000
export VAL_STEP=500
export NUM_INFERENCE_STEPS=20  # Number of denoising steps during validation

# output dir
export OUTPUT_DIR="output/finetune-sdaig-pred-bsz${TOTAL_BSZ}/"

accelerate launch --config_file=accelerate_configs/$CUDA.yaml --mixed_precision="fp16" \
  finetune_sdaig_pred.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_config_path=$DATASET_CONFIG \
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
  --resume_from_checkpoint="latest" \
  --lora_rank=$LORA_RANK \
  --lora_alpha=$LORA_ALPHA \
  --lora_dropout=$LORA_DROPOUT