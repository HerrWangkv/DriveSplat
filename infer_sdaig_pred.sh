
export CUDA=0

export CHECKPOINT_DIR="output/train-sdaig-pred-bsz8/checkpoint-15500"
export OUTPUT_DIR="output/Depth_G_Infer_Pred"
export DATASET_CONFIG="data/NuScenes/sdaig.yaml"

CUDA_VISIBLE_DEVICES=$CUDA python infer_pred.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --dataset_config=$DATASET_CONFIG \
        --prediction_type="sample" \
        --seed=42 \
        --half_precision \
        --output_dir=$OUTPUT_DIR \
        --disparity 
        # --processing_res=0 # Defualt: 768. To obtain more fine-grained results, you can set `--processing_res=0` (original resolution) or a higher resolution. 