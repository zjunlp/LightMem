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
BATCH_SIZE=16
MODALITIES=("image" "video" "visdoc")
DATA_BASEDIR="~/data/vlm2vec_eval"
OUTPUT_BASEDIR="~/exps/vlm2vec/"


# ==> Define models and their base output paths here
# Format: "MODEL_NAME;BASE_OUTPUT_PATH"
declare -a MODEL_SPECS
MODEL_SPECS+=( "VLM2Vec/VLM2Vec-V2.0;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B" )
MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-2B-Instruct" )
MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct" )
MODEL_SPECS+=( "code-kunkun/LamRA-Ret;lamra;$OUTPUT_BASEDIR/LamRA-Ret" )
MODEL_SPECS+=( "vidore/colpali-v1.3;colpali;$OUTPUT_BASEDIR/colpali-v1.3" )


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

    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval.py \
      --pooling eos \
      --normalize true \
      --per_device_eval_batch_size $BATCH_SIZE \
      --model_backbone \"$MODEL_BACKBONE\" \
      --model_name \"$MODEL_NAME\" \
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
