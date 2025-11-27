# LightMem Evaluation Scripts

Evaluation scripts for building and searching memory collections on the LoCoMo dataset.

## Overview

This repository contains two main scripts:

- **`add_locomo.py`**: Build memory collections from conversation data
- **`search_locomo.py`**: Retrieve memories and evaluate QA performance

## Quick Start

### Step 1: Build Memory Collections

Process conversations and build memory collections with vector embeddings:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python add_locomo.py \
    > build_memory.log 2>&1 &
```

**Configuration**: Edit the configuration section in `add_locomo.py` before running:
- API keys and models
- Model paths (LLMlingua, embedding models)
- Dataset path
- Output directories
- Number of parallel workers

**Output**: 
- `./qdrant_pre_update/` - Memory state before updates
- `./qdrant_post_update/` - Memory state after updates
- `./logs/` - Detailed logs for each sample

### Step 2: Evaluate with Vector Retrieval

Retrieve relevant memories and evaluate QA performance:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python search_locomo.py \
    --dataset /path/to/locomo10.json \
    --qdrant-dir ./qdrant_post_update \
    --output-dir ./results/evaluation_combined_60 \
    --total-limit 60 \
    --retrieval-mode combined \
    --embedder huggingface \
    --embedding-model-path /path/to/all-MiniLM-L6-v2 \
    --llm-api-key sk-xxx \
    --llm-base-url xxx \
    --llm-model gpt-4o-mini \
    --judge-api-key sk-xxx \
    --judge-base-url xxx \
    --judge-model gpt-4o-mini \
    > evaluation.log 2>&1 &
```

**Key Arguments**:
- `--retrieval-mode`: `combined` (top-k across speakers) or `per-speaker` (top-k per speaker)
- `--total-limit`: Number of memories to retrieve (for `combined` mode)
- `--limit-per-speaker`: Number of memories per speaker (for `per-speaker` mode)
- `--embedder`: `huggingface` or `openai`

**Note**: We uses `--retrieval-mode combined` with `--total-limit 60`.

**Output**:
- `./results/sample_*.json` - Per-sample results
- `./results/summary.json` - Aggregate metrics and statistics

## Utilities

- **`retrievers.py`**: Vector retrieval utilities (QdrantEntryLoader, VectorRetriever)
- **`llm_judge.py`**: LLM-based answer evaluation.
- **`prompt.py`**: Prompt templates for answer generation.

These modules are imported by `search_locomo.py` and can be reused in other evaluation scripts.