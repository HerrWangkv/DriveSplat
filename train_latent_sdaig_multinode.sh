#!/bin/bash

# The data directories are now mounted by the SLURM script
# Just verify they exist
if [ ! -d "/data/raw" ]; then
    echo "Error: /data/raw not found. SquashFS mount may have failed."
    exit 1
fi

if [ ! -d "/data/cache" ]; then
    echo "Warning: /data/cache not found. Cache SquashFS may not be available."
fi

echo "Data directories are available:"
ls -la /data/

export MODEL_NAME="jingheya/lotus-depth-g-v2-1-disparity"

# training dataset
export DATASET_CONFIG_PATH="data/NuScenes/latent_sdaig.yaml"

# training configs
export BATCH_SIZE=1
export GAS=1
export TOTAL_BSZ=$(($BATCH_SIZE * 8 * $GAS))  # 8 GPUs total across 2 nodes (4 per node)

# model configs for multi-step training
export MAX_TIMESTEPS=1000
export VAL_STEP=1000
export NUM_INFERENCE_STEPS=20  # Number of denoising steps during validation

# output dir
export OUTPUT_DIR="output/train-latent-sdaig-bsz${TOTAL_BSZ}/"

# Set machine rank based on hostname and head node
current_hostname=$(hostname -s)
echo "Current hostname: $current_hostname"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Master address: $MASTER_ADDR"

# The head node (master) should always have rank 0
# Get head node IP and determine which node is the head node
head_node_ip="$MASTER_ADDR"

# Check if current node is the head node by comparing IPs
current_ip=$(hostname -I | awk '{print $1}')
echo "Current IP: $current_ip"
echo "Head node IP: $head_node_ip"

if [[ "$current_ip" == "$head_node_ip" ]]; then
    export MACHINE_RANK=0
    echo "This node is the HEAD NODE (master)"
else
    export MACHINE_RANK=1
    echo "This node is a WORKER NODE"
fi

echo "Current hostname: $(hostname -s)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Machine rank: $MACHINE_RANK"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_NODEID: $SLURM_NODEID"

# Network connectivity test removed (nc command not available in container)
# The actual distributed training uses PyTorch's NCCL backend for communication

# Use accelerate launch with simplified multi-node configuration
# Removed timeout to let training run for full duration
accelerate launch \
  --multi_gpu \
  --mixed_precision="fp16" \
  --num_machines=2 \
  --num_processes=8 \
  --machine_rank=$MACHINE_RANK \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  train_latent_sdaig.py \
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
