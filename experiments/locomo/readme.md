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

<span id='results'/>

## 📊 Results on LoCoMo

backbone: `gpt-4o-mini`, judge model: `gpt-4o-mini` & `qwen2.5-32b-instruct`

| Method            | ACC(%) gpt-4o-mini | ACC(%) qwen2.5-32b-instruct  | F1 Score | BLEU-1             | Memory-Con Tokens(k) Total | QA Tokens(k) total | Total(k)     | Calls  | Runtime(s) total |
|-------------------|--------------------|------------------------------|----------|------------------- |----------------------------|--------------------|--------------|--------|------------------|
| LightMem(512,0.7) | 71.95              | 73.90                        | 47.75    |36.16               |   997.61                   | 4,008.243          | 5,005.851    | 415    | 12,831          |
| LightMem(768,0.7) | 70.26              | 72.40                        | 47.18    |36.12               |  764.78                    | 3,958.228          | 4,723.005    | 255    | 11,643          |
| LightMem(768,0.8) | 72.99              | 74.35                        | 47.77    |36.57               |  851.83                    | 4,012.034          | 4,863.900    | 298    | 12,423          |

backbone: `qwen3-30b-a3b-instruct-2507`, judge model: `gpt-4o-mini` & `qwen2.5-32b-instruct`

| Method             | ACC(%) gpt-4o-mini | ACC(%) qwen2.5-32b-instruct | F1 Score | BLEU-1             | Memory-Con Tokens(k) Total | QA Tokens(k) total | Total(k)     | Calls  | Runtime(s) total |
|-------------------|--------------------|------------------------------|----------|------------------- |-----------------------------|---------------------|--------------|--------|------------------|
| LightMem(768,0.4) | 64.09              | 60.84                        | 38.22    |31.15               | 781.908                     | 5,151.393           | 5,878.075    | 174    | 9,638            |
| LightMem(768,0.6) | 71.36              | 69.03                        | 43.18    |35.37               | 998.728                     | 5,220.252           | 6,218.980    | 291    | 10,541           |
| LightMem(1024,0.8)| 72.60              | 71.36                        | 44.78    |37.06               | 1,084.465                   | 5,304.471           | 6,388.936    | 320    | 13,075           |


### Details

backbone: `gpt-4o-mini`, judge model: `gpt-4o-mini` & `qwen2.5-32b-instruct`

| Method             | Summary Tokens(k) In | Summary Tokens(k) Out | Update Tokens(k) In | Update Tokens(k) Out | QA Tokens(k) In | QA Tokens(k) Out | Runtime(s) mem-con | Runtime(s) qa |
|-------------------|-----------------------|------------------------|----------------------|-----------------------|------------------|-------------------|----------------------|----------------|
| LightMem(512,0.7) | 731.89                | 201.29                 | 60.45                | 3.97                  | 3,997.984        | 10.259            | 8,484                | 4,347          |
| LightMem(768,0.7) | 555.36                | 170.24                 | 36.85                | 2.32                  | 3,948.124        | 10.104            | 7,388                | 4,265          |
| LightMem(768,0.8) | 628.20                | 179.53                 | 41.38                | 2.76                  | 4,001.759        | 10.275            | 8,153                | 4,270          |

backbone: `qwen3-30b-a3b-instruct-2507`, judge model: `gpt-4o-mini` & `qwen2.5-32b-instruct`

| Method             | Summary Tokens(k) In | Summary Tokens(k) Out | Update Tokens(k) In | Update Tokens(k) Out | QA Tokens(k) In | QA Tokens(k) Out | Runtime(s) mem-con | Runtime(s) qa |
|-------------------|-----------------------|------------------------|----------------------|-----------------------|------------------|-------------------|----------------------|----------------|
| LightMem(768,0.4) | 430.572               | 296.110                | 51.026               | 4.200                 | 5,132.643        | 18.750            | 7,309                | 2,329          |
| LightMem(768,0.6) | 566.803               | 341.381                | 83.135               | 7.409                 | 5,201.980        | 18.272            | 8,157                | 2,384          |
| LightMem(1024,0.8)| 613.820               | 363.293                | 98.593               | 8.759                 | 5,288.685        | 15.786            | 10,794               | 2,281          |

#### Performance metrics
backbone: `gpt-4o-mini`, judge model: `gpt-4o-mini`

| Method | Overall ↑ | Multi | Open | Single | Temp |F1 Score | BLEU-1 | 
| :---   | :---:     | :---: | :---:| :---:  | :---:| :---:   | :---:|
| LightMem(512,0.7)| 71.95 | 62.41 | 44.79 | 77.41 | 74.14 | 47.75    | 36.16    |
| LightMem(768,0.7)| 70.26 | 62.06 | 42.71 | 74.67 | 74.14 | 47.18    | 36.12    |
| LightMem(768,0.8)| 72.99 | 67.02 | 45.83 | 76.81 | 76.32 | 47.77    | 36.57    |

backbone: `gpt-4o-mini`, judge model: `qwen2.5-32b-instruct`

| Method | Overall ↑ | Multi | Open | Single | Temp |F1 Score | BLEU-1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---:   | :---:|
| LightMem(512,0.7)| 73.90 | 69.15 | 50.00 | 78.00 | 74.45 | 47.75    | 36.16    |
| LightMem(768,0.7)| 72.40 | 64.54 | 43.75 | 77.17 | 75.39 | 47.18    | 36.12    |
| LightMem(768,0.8)| 74.35 | 68.79 | 47.92 | 78.24 | 76.95 | 47.77    | 36.57    |

backbone: `qwen3-30b-a3b-instruct-2507`, judge model: `gpt-4o-mini`

| Method | Overall ↑ | Multi | Open | Single | Temp |F1 Score | BLEU-1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---:   | :---:|
| LightMem(768,0.4)| 64.09 | 63.12 | 45.83 | 72.29 | 48.91 | 38.22    | 31.15    |
| LightMem(768,0.6)| 71.36 | 70.57 | 60.42 | 79.19 | 54.83 | 43.18    | 35.37    |
| LightMem(1024,0.8)|72.60 | 72.34 | 50.00 | 82.16 | 54.52 | 44.78    | 37.06    |

backbone: `qwen3-30b-a3b-instruct-2507`, judge model: `qwen2.5-32b-instruct`

| Method | Overall ↑ | Multi | Open | Single | Temp |F1 Score | BLEU-1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---:   | :---:|
| LightMem(768,0.4)| 60.84 | 59.22 | 42.68 | 69.92 | 42.68 | 38.22    | 31.15    |
| LightMem(768,0.6)| 69.03 | 67.38 | 60.42 | 78.48 | 48.29 | 43.18    | 35.37    |
| LightMem(1024,0.8)|71.36 | 68.09 | 52.08 | 82.76 | 50.16 | 44.78    | 37.06    |

