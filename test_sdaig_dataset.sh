#!/bin/bash

mkdir -p /data/raw
mkdir -p /data/cache
squashfuse /data/nuscenes.sqfs /data/raw/
squashfuse /data/cache.sqfs /data/cache

python test_sdaig_dataset.py