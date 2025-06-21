
export CUDA=0

export CHECKPOINT_DIR="output/train-sdaig-bsz8/checkpoint-3500"
export OUTPUT_DIR="output/Depth_G_Infer"
export DATASET_CONFIG="data/NuScenes/sdaig.yaml"

CUDA_VISIBLE_DEVICES=$CUDA python infer.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --dataset_config=$DATASET_CONFIG \
        --prediction_type="sample" \
        --seed=42 \
        --half_precision \
        --output_dir=$OUTPUT_DIR \
        --disparity 
        # --processing_res=0 # Defualt: 768. To obtain more fine-grained results, you can set `--processing_res=0` (original resolution) or a higher resolution. 