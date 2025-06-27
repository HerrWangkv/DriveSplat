#!/bin/bash
#SBATCH --job-name=drivesplat_train_single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/train_job_single_%j.out
#SBATCH --error=logs/train_job_single_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
# export WANDB_API_KEY="${WANDB_API_KEY}"

# SquashFS configuration
export DATA_PATH="/hkfs/work/workspace/scratch/xw2723-nuscenes"
export NUSCENES_SQFS="$DATA_PATH/nuscenes.sqfs"
export CACHE_SQFS="$DATA_PATH/cache.sqfs"
export MOUNT_PATH_RAW="/dev/shm/$(whoami)/sqsh/nuscenes"
export MOUNT_PATH_CACHE="/dev/shm/$(whoami)/sqsh/cache"

# SquashFS mount/unmount functions
unmount_squashfuse() {
    [ -d "$MOUNT_PATH_RAW" ] && fusermount3 -u "$MOUNT_PATH_RAW" 2>/dev/null || true
    [ -d "$MOUNT_PATH_CACHE" ] && fusermount3 -u "$MOUNT_PATH_CACHE" 2>/dev/null || true
    rm -rf "$MOUNT_PATH_RAW" "$MOUNT_PATH_CACHE"
}

mount_squashfuse() {
    # Clean up any existing mounts
    [ -d "$MOUNT_PATH_RAW" ] && fusermount3 -u "$MOUNT_PATH_RAW" 2>/dev/null || true
    [ -d "$MOUNT_PATH_CACHE" ] && fusermount3 -u "$MOUNT_PATH_CACHE" 2>/dev/null || true
    rm -rf "$MOUNT_PATH_RAW" "$MOUNT_PATH_CACHE"

    # Create mount directories
    mkdir -p "$MOUNT_PATH_RAW" "$MOUNT_PATH_CACHE"
    chmod 700 "$MOUNT_PATH_RAW" "$MOUNT_PATH_CACHE"

    # Register cleanup handler
    trap unmount_squashfuse EXIT SIGINT SIGTERM SIGCONT

    # Mount SquashFS files
    if [ -f "$NUSCENES_SQFS" ]; then
        squashfuse_ll "$NUSCENES_SQFS" "$MOUNT_PATH_RAW" || exit 1
        echo "Mounted $NUSCENES_SQFS at $MOUNT_PATH_RAW"
    else
        echo "Warning: $NUSCENES_SQFS not found"
    fi

    if [ -f "$CACHE_SQFS" ]; then
        squashfuse_ll "$CACHE_SQFS" "$MOUNT_PATH_CACHE" || exit 1
        echo "Mounted $CACHE_SQFS at $MOUNT_PATH_CACHE"
    else
        echo "Warning: $CACHE_SQFS not found"
    fi
}

echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Nodes: $SLURM_JOB_NODELIST"

# Mount SquashFS
mount_squashfuse

echo "SquashFS mounts are ready"

# Run the training script (single node with 4 GPUs)
apptainer exec --nv \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --bind /home/hk-project-p0023969/xw2723/test/DriveSplat:/workspace \
  --bind "${MOUNT_PATH_RAW}:/data/raw" \
  --bind "${MOUNT_PATH_CACHE}:/data/cache" \
  --pwd /workspace \
  drivesplat.sif \
  bash train_sdaig_pred_multistep_4gpu.sh
