#!/bin/bash

mkdir -p /data/raw
mkdir -p /data/cache
fusermount -u /data/raw || true
fusermount -u /data/cache || true
squashfuse /data/nuscenes.sqfs /data/raw/
squashfuse /data/cache.sqfs /data/cache

python test_sdaig_dataset.py