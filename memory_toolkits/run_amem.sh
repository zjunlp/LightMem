#!/bin/bash

# A-MEM 启动脚本
# 使用方法: ./run_amem.sh [dataset_path] [config_path]
export OPENAI_API_BASE="https://api.gpts.vin/v1"
export OPENAI_API_KEY="sk-96TZyg8iXZGGBWUp8osOpf0YhNJ9t3ag0HY4Gk5V6uRD9IwQ"
# 设置默认值
DEFAULT_DATASET_PATH="/disk/disk_4T_2/jiangziyan1/datasets/longmemeval/longmemeval_s.json"
DEFAULT_CONFIG_PATH="/disk/disk_4T_2/jiangziyan1/LightMem/memory_toolkits/configs/amem_config.json"
DEFAULT_NUM_WORKERS=5
DEFAULT_SEED=42
DEFAULT_SAMPLE_SIZE=100

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用默认参数启动A-MEM..."
    DATASET_PATH=$DEFAULT_DATASET_PATH
    CONFIG_PATH=$DEFAULT_CONFIG_PATH
elif [ $# -eq 1 ]; then
    DATASET_PATH=$1
    CONFIG_PATH=$DEFAULT_CONFIG_PATH
elif [ $# -eq 2 ]; then
    DATASET_PATH=$1
    CONFIG_PATH=$2
else
    echo "使用方法: $0 [dataset_path] [config_path]"
    echo "参数说明:"
    echo "  dataset_path: 数据集路径 (默认: $DEFAULT_DATASET_PATH)"
    echo "  config_path:  配置文件路径 (默认: $DEFAULT_CONFIG_PATH)"
    exit 1
fi

echo "🚀 启动A-MEM评估..."
echo "📁 数据集路径: $DATASET_PATH"
echo "⚙️  配置文件: $CONFIG_PATH"
echo "🔧 工作线程数: $DEFAULT_NUM_WORKERS"
echo "🎲 随机种子: $DEFAULT_SEED"
echo "📊 样本大小: $DEFAULT_SAMPLE_SIZE"
echo ""

# 运行评估脚本
python evaluation.py \
    --memory-type "A-MEM" \
    --dataset-type "LongMemEval" \
    --dataset-path "$DATASET_PATH" \
    --config_path "$CONFIG_PATH" \
    --num-workers $DEFAULT_NUM_WORKERS \
    --seed $DEFAULT_SEED \
    --sample-size $DEFAULT_SAMPLE_SIZE \
    --token-cost-save-filename "amem_token_cost"

echo ""
echo "✅ A-MEM评估完成！" 