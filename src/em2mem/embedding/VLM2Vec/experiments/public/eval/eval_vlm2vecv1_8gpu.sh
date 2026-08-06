#!/bin/bash

echo "==> Environment"
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

cd projects/VLM2Vec/ || exit

# ==============================================================================
# Configuration
# ==============================================================================
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
BATCH_SIZE=32
MODALITIES=("image" "video" "visdoc")
DATA_BASEDIR="~/data/vlm2vec_eval"
OUTPUT_BASEDIR="~/exps/vlm2vec/"

# ==> Define models and their base output paths here
# Format: "MODEL_NAME;BASE_OUTPUT_PATH"
declare -a MODEL_SPECS
MODEL_SPECS+=( "TIGER-Lab/VLM2Vec-Qwen2VL-2B;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V1-Qwen2VL-2B" )
MODEL_SPECS+=( "TIGER-Lab/VLM2Vec-Qwen2VL-7B;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V1-Qwen2VL-7B" )


# ==============================================================================
# Main Execution Loop
# ==============================================================================
# Loop through each model specification
for spec in "${MODEL_SPECS[@]}"; do
  # Parse the model name and base output path from the spec string
  IFS=';' read -r MODEL_NAME MODEL_BACKBONE BASE_OUTPUT_PATH <<< "$spec"

  echo "================================================="
  echo "🚀 Processing Model: $MODEL_NAME"
  echo "================================================="

  # Loop through each modality for the current model
  for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="experiments/release/eval/$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    # Ensure the output directory exists
    mkdir -p "$OUTPUT_PATH"

    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=8 --master_port=2233 --max_restarts=0 eval.py \
      --pooling eos \
      --normalize true \
      --per_device_eval_batch_size $BATCH_SIZE \
      --model_backbone \"$MODEL_BACKBONE\" \
      --model_name \"$MODEL_NAME\" \
      --resize_use_processor false \
      --image_resolution high \
      --dataset_config \"$DATA_CONFIG_PATH\" \
      --encode_output_path \"$OUTPUT_PATH\" \
      --data_basedir \"$DATA_BASEDIR\""

    echo "  - Executing command..."
    # echo "$cmd" # Uncomment for debugging the exact command
    eval "$cmd"
    echo "  - Done."
    echo "-------------------------------------------------"
  done
done

echo "✅ All jobs completed."
