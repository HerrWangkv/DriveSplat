#!/bin/bash

docker run -it --rm --gpus all --name sdaig \
  --privileged \
  --shm-size=8g \
  -v /storage_local/kwang/repos/DriveSplat:/workspace \
  -v /mrtstorage/datasets/public/nuscenes.sqfs:/data/nuscenes.sqfs \
  -v /mrtstorage/users/kwang/nuscenes_cache_10hz.sqfs:/data/cache.sqfs \
  -v /storage_local/kwang/nuscenes:/data \
  -w /workspace \
  drivesplat:latest  \
  bash test_sdaig_dataset.sh \