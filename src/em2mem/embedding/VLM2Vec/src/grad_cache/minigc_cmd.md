

#### Prerequisite
```bash
ENV_PATH=/export/share/ruimeng/env/anaconda/envs/llm/bin/ninja
export PATH="${ENV_PATH}/:$PATH"

export NCCL_DEBUG=WARN
export HF_DATASETS_CACHE=/export/xgen-embedding/data/.hfdata_cache
export TRANSFORMERS_CACHE=/export/xgen-embedding/data/.hfmodel_cache/
export TOKENIZERS_PARALLELISM=true
export WANDB_DISABLED=false
export WANDB_PROJECT=mini-gradcache
export WANDB_API_KEY=local-d64a4127e8d4a1782aedbb72e76080b3dfbf89dd
export WANDB_BASE_URL=https://salesforceairesearch.wandb.io
```

```bash
# gpu0-3, DDP4-bs4096-accum4, 29922MB, hang at epoch34
export EXP_NAME=GC-4gpu-bs4096-accum16-step10k
export EXP_DIR=/export/xgen-embedding/runs/ruimeng/minimal_gc/$EXP_NAME
export WANDB_DIR=$EXP_DIR/wandb
export WANDB_NAME=$EXP_NAME
export WORLD_SIZE=4
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/*
cd /export/home/project/search/xgen-embedding/
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=4403 --max_restarts=0 mini_gc.py --model_name_or_path bert-base-uncased --output_dir $EXP_DIR --q_len 128 --d_len 256 --batch_size 4096 --chunk_sizes 256 2>&1 | tee $EXP_DIR/train.log


# gpu0-3, DDP4-bs256-accum4, 11818MB
export EXP_NAME=GC-4gpu-bs256-accum4-step10k
export EXP_DIR=/export/xgen-embedding/runs/ruimeng/minimal_gc/$EXP_NAME
export WANDB_DIR=$EXP_DIR/wandb
export WANDB_NAME=$EXP_NAME
export WORLD_SIZE=4
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/*
cd /export/home/project/search/xgen-embedding/
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=4403 --max_restarts=0 mini_gc.py --model_name_or_path bert-base-uncased --output_dir $EXP_DIR --q_len 128 --d_len 256 --batch_size 64 --chunk_sizes 16 2>&1 | tee $EXP_DIR/train.log



# gpu45, DDP2-bs256-accum2, 15742MB
export EXP_NAME=GC-2gpu-bs256-accum2-step10k
export EXP_DIR=/export/xgen-embedding/runs/ruimeng/minimal_gc/$EXP_NAME
export WANDB_DIR=$EXP_DIR/wandb
export WANDB_NAME=$EXP_NAME
export WORLD_SIZE=1
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/*
cd /export/home/project/search/xgen-embedding/
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=2245 --max_restarts=0 mini_gc.py --model_name_or_path bert-base-uncased --output_dir $EXP_DIR --q_len 128 --d_len 256 --batch_size 128 --chunk_sizes 64 2>&1 | tee $EXP_DIR/train.log


# gpu6, bs256-accum4, 9GB
export EXP_NAME=GC-1gpu-bs256-accum4-step10k
export EXP_DIR=/export/xgen-embedding/runs/ruimeng/minimal_gc/$EXP_NAME
export WANDB_DIR=$EXP_DIR/wandb
export WANDB_NAME=$EXP_NAME
export WORLD_SIZE=1
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/*
cd /export/home/project/search/xgen-embedding/
CUDA_VISIBLE_DEVICES=6 python -m mini_gc --model_name_or_path bert-base-uncased --output_dir $EXP_DIR --q_len 128 --d_len 256 --batch_size 256 --chunk_sizes 64 2>&1 | tee $EXP_DIR/train.log


# gpu6, bs256-accum2, 18GB
export EXP_NAME=GC-1gpu-bs256-accum2-step10k
export EXP_DIR=/export/xgen-embedding/runs/ruimeng/minimal_gc/$EXP_NAME
export WANDB_DIR=$EXP_DIR/wandb
export WANDB_NAME=$EXP_NAME
export WORLD_SIZE=1
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/*
cd /export/home/project/search/xgen-embedding/
CUDA_VISIBLE_DEVICES=6 python -m mini_gc --model_name_or_path bert-base-uncased --output_dir $EXP_DIR --q_len 128 --d_len 256 --batch_size 256 --chunk_sizes 128 2>&1 | tee $EXP_DIR/train.log


# gpu7, bs256-accum1, 38012MB
export EXP_NAME=GC-1gpu-bs256-accum1-step10k-baseline
export EXP_DIR=/export/xgen-embedding/runs/ruimeng/minimal_gc/$EXP_NAME
export WANDB_DIR=$EXP_DIR/wandb
export WANDB_NAME=$EXP_NAME
export WORLD_SIZE=1
mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/*
cd /export/home/project/search/xgen-embedding/
CUDA_VISIBLE_DEVICES=7 python -m mini_gc --model_name_or_path bert-base-uncased --output_dir $EXP_DIR --q_len 128 --d_len 256 --batch_size 256 --chunk_sizes -1 2>&1 | tee $EXP_DIR/train.log
```
