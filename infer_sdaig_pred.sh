#!/bin/bash

mkdir -p /data/raw
mkdir -p /data/cache
squashfuse /data/nuscenes.sqfs /data/raw/
squashfuse /data/cache.sqfs /data/cache

export CUDA=0

export CHECKPOINT_DIR="output/box_depth&render_w_moving_objects/final"
export OUTPUT_DIR="output/infer-box_depth&render_w_moving_objects"
export DATASET_CONFIG="data/NuScenes/sdaig.yaml"
export MAX_TIMESTEPS=1000
export NUM_INFERENCE_STEPS=20

CUDA_VISIBLE_DEVICES=$CUDA python infer_sdaig_pred.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --dataset_config=$DATASET_CONFIG \
        --prediction_type="sample" \
        --seed=42 \
        --half_precision \
        --output_dir=$OUTPUT_DIR \
        --multi_step \
        --max_timesteps=$MAX_TIMESTEPS \
        --num_inference_steps=$NUM_INFERENCE_STEPS \
        --create_videos \
        --create_comparison_videos 
        # --processing_res=0 # Defualt: 768. To obtain more fine-grained results, you can set `--processing_res=0` (original resolution) or a higher resolution. 