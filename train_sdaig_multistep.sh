#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-2-base"

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
export OUTPUT_DIR="output/train-sdaig-multistep-bsz${TOTAL_BSZ}/"

accelerate launch --config_file=accelerate_configs/$CUDA.yaml --mixed_precision="fp16" \
  --main_process_port="13224" \
  train_sdaig.py \
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
  --checkpoints_total_limit=5 \
  --report_to="wandb"
