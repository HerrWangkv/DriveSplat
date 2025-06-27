#!/bin/bash
#SBATCH --job-name=drivesplat_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=dev_accelerated
#SBATCH --output=logs/train_job_%j.out
#SBATCH --error=logs/train_job_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
# export WANDB_API_KEY="${WANDB_API_KEY}"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_TIMEOUT=3600
export NCCL_TREE_THRESHOLD=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ALGO=Ring
export NCCL_IGNORE_CPU_AFFINITY=1

# SquashFS configuration
export DATA_PATH="/hkfs/work/workspace/scratch/xw2723-nuscenes"
export NUSCENES_SQFS="$DATA_PATH/nuscenes.sqfs"
export CACHE_SQFS="$DATA_PATH/cache.sqfs"
export MOUNT_PATH_RAW="/dev/shm/$(whoami)/sqsh/nuscenes"
export MOUNT_PATH_CACHE="/dev/shm/$(whoami)/sqsh/cache"

# SquashFS mount/unmount functions
unmount_squashfuse() {
    # Do nothing on tasks with node-local rank other than 0
    ((SLURM_LOCALID)) && return 0
    [ -d "$MOUNT_PATH_RAW" ] && fusermount3 -u "$MOUNT_PATH_RAW" 2>/dev/null || true
    [ -d "$MOUNT_PATH_CACHE" ] && fusermount3 -u "$MOUNT_PATH_CACHE" 2>/dev/null || true
    rm -rf "$MOUNT_PATH_RAW" "$MOUNT_PATH_CACHE"
}
export -f unmount_squashfuse

mount_squashfuse() {
    # Do nothing on tasks with node-local rank other than 0
    ((SLURM_LOCALID)) && return 0

    # Clean up any existing mounts
    [ -d "$MOUNT_PATH_RAW" ] && fusermount3 -u "$MOUNT_PATH_RAW" 2>/dev/null || true
    [ -d "$MOUNT_PATH_CACHE" ] && fusermount3 -u "$MOUNT_PATH_CACHE" 2>/dev/null || true
    rm -rf "$MOUNT_PATH_RAW" "$MOUNT_PATH_CACHE"

    # Create mount directories
    mkdir -p "$MOUNT_PATH_RAW" "$MOUNT_PATH_CACHE"
    chmod 700 "$MOUNT_PATH_RAW" "$MOUNT_PATH_CACHE"

    # Register cleanup handler
    trap 'bash -c unmount_squashfuse' EXIT SIGINT SIGTERM SIGCONT

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

    # Keep the mount process alive
    while true; do
        sleep 90000  # 25 hours
    done
}
export -f mount_squashfuse

wait_for_mount() {
    # Get the process ID of the most recent mount process
    mount_pid="$(pgrep -n -f -u "$(whoami)" -- ' -c mount_squashfuse$' 2>/dev/null || echo "")"
    if [ -n "$mount_pid" ]; then
        # Wait for mount to complete
        while ps -p "$mount_pid" > /dev/null 2>&1 && ! mountpoint -q "$MOUNT_PATH_RAW" 2>/dev/null; do
            sleep 1
        done
    fi
}
export -f wait_for_mount

# Get the list of nodes
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | head -n1)

# Validate that we got a proper IP
if [[ ! $head_node_ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Could not get valid IP address for head node. Got: $head_node_ip"
    exit 1
fi

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

# Set master node and port for distributed training
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

# Mount the SquashFS files (in background, with resource overlap)
srun --overlap bash -c mount_squashfuse &

# Wait for mount to complete on all nodes
srun bash -c wait_for_mount

echo "SquashFS mounts are ready"

# Run the training script on all nodes
srun apptainer exec --nv \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --env MASTER_ADDR="${MASTER_ADDR}" \
  --env MASTER_PORT="${MASTER_PORT}" \
  --env SLURM_PROCID="${SLURM_PROCID}" \
  --env SLURM_LOCALID="${SLURM_LOCALID}" \
  --env SLURM_NODEID="${SLURM_NODEID}" \
  --env SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES}" \
  --env SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST}" \
  --env NCCL_DEBUG="${NCCL_DEBUG}" \
  --env NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
  --env NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}" \
  --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  --env NCCL_TIMEOUT="${NCCL_TIMEOUT}" \
  --env NCCL_TREE_THRESHOLD="${NCCL_TREE_THRESHOLD}" \
  --env NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}" \
  --env NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING}" \
  --env NCCL_ALGO="${NCCL_ALGO}" \
  --env NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY}" \
  --env MOUNT_PATH_RAW="${MOUNT_PATH_RAW}" \
  --env MOUNT_PATH_CACHE="${MOUNT_PATH_CACHE}" \
  --bind /home/hk-project-p0023969/xw2723/test/DriveSplat:/workspace \
  --bind "${MOUNT_PATH_RAW}:/data/raw" \
  --bind "${MOUNT_PATH_CACHE}:/data/cache" \
  --pwd /workspace \
  drivesplat.sif \
  bash train_sdaig_pred_multistep_multinode.sh
