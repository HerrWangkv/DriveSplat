
export CUDA=0

export CHECKPOINT_DIR="jingheya/lotus-depth-g-v2-1-disparity"
export OUTPUT_DIR="output/Depth_G_Infer"

CUDA_VISIBLE_DEVICES=$CUDA python infer.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --prediction_type="sample" \
        --seed=42 \
        --half_precision \
        --output_dir=$OUTPUT_DIR \
        --disparity 
        # --processing_res=0 # Defualt: 768. To obtain more fine-grained results, you can set `--processing_res=0` (original resolution) or a higher resolution. 