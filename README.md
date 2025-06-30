## Docker Build
```bash
docker build -t drivesplat .
```

## Apptainer Build
```bash
apptainer build drivesplat.sif docker-daemon://drivesplat:latest
```
or
```bash
apptainer pull drivesplat.sif docker://kaiwenwangkit/drivesplat:20250628
```

## Running the Container
### Apptainer



squashfuse /data/nuscenes.sqfs /data/raw/
squashfuse /data/cache.sqfs /data/cache
export WANDB_API_KEY="8c922d3dd66cfe0107acda9d965f7247cb11ae83"
export NCCL_P2P_LEVEL=NVL
bash train_sdaig_pred_multistep.sh