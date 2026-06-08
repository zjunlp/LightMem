# EM²Mem Evaluation Scripts

Evaluation scripts for building and evaluating event-centric multimodal memory on the EgoLifeQA benchmark.

## Overview

This directory contains the EgoLife pipeline for **EM²Mem**:

- **`scripts/1_setup.sh`**: Create the EgoLife `uv` environment and optionally download EgoLife data
- **`scripts/2_preprocess.sh`**: Translate dense captions and generate synchronized EgoLife records
- **`scripts/3_build_multimodal_memory_cell.sh`**: Build event-centric multimodal memory cells, temporal context views, and episodic graphs
- **`scripts/4_build_semantic_graph.sh`**: Extract and consolidate semantic memory from episodic evidence
- **`scripts/5_precompute_rag_embeddings.sh`**: Precompute dense episodic and semantic RAG caches
- **`scripts/6_eval.sh`**: Evaluate EM²Mem on EgoLifeQA

EM²Mem uses event records as the primary multimodal evidence interface. In EgoLife, visual evidence is loaded from `output/metadata/multimodal_memory_cell/<PERSON>/<PERSON>_record.json`, which contains event-aligned keyframe paths, keyframe captions, visual objects, scene fields, and timestamps. The main EgoLife pipeline does not require `visual_embeddings.pkl`.

## Quick Start

### Step 1: Set Up the EgoLife Environment

Create an isolated `uv` environment for EgoLife:

```bash
cd experiments/egolife
bash scripts/1_setup.sh --skip-data
source .venv/bin/activate
```

To download EgoLife through the setup script, omit `--skip-data` or specify a data directory:

```bash
bash scripts/1_setup.sh --data-dir data/EgoLife
```

Prepare API credentials either as environment variables or with an explicit `.env` file passed to later scripts through `--env-file`:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-5-mini"
```

### Step 2: Preprocess EgoLife

Translate DenseCaption files and generate synchronized EgoLife records:

```bash
bash scripts/2_preprocess.sh \
    --person A1_JAKE \
    --base-dir data/EgoLife
```

**Output**:
- `data/EgoLife/EgoLifeCap/DenseCaption/translated/<PERSON>/`
- `data/EgoLife/EgoLifeCap/Sync/`

### Step 3: Build Multimodal Memory Cells

Build event-centric multimodal memory cells, multi-scale temporal context views, and episodic graphs:

```bash
bash scripts/3_build_multimodal_memory_cell.sh \
    --person A1_JAKE \
    --base-dir data/EgoLife \
    --output-dir output \
    --model gpt-5-mini
```

**Output**:
- `output/metadata/multimodal_memory_cell/<PERSON>/<PERSON>_record.json`
- `output/metadata/multimodal_memory_cell/<PERSON>/temporal_context_views/`
- `output/metadata/multimodal_memory_cell/<PERSON>/{30sec,3min,10min,1h}/episodic_triplets_*.json`
- `output/metadata/multimodal_memory_cell/<PERSON>/{30sec,3min,10min,1h}/episodic_graph_*.json`

### Step 4: Build Semantic Memory

Extract candidate semantic triples and consolidate them into the semantic graph:

```bash
bash scripts/4_build_semantic_graph.sh \
    --person A1_JAKE \
    --output-dir output \
    --model gpt-5-mini
```

**Output**:
- `output/metadata/semantic_graph/<PERSON>/semantic_candidates_gpt-5-mini.json`
- `output/metadata/semantic_graph/<PERSON>/semantic_graph_gpt-5-mini.json`

### Step 5: Precompute RAG Embeddings

Precompute dense episodic and semantic caches before parallel evaluation:

```bash
bash scripts/5_precompute_rag_embeddings.sh \
    --person A1_JAKE \
    --output-dir output \
    --text-embedding-model Qwen/Qwen3-Embedding-4B \
    --gpu 0
```

**Output**:
- `.cache/episodic_dense/<PERSON>/`
- `.cache/semantic_dense/<PERSON>/`

### Step 6: Evaluate on EgoLifeQA

Run EM²Mem evaluation:

```bash
bash scripts/6_eval.sh \
    --person A1_JAKE \
    --data-dir data/EgoLife \
    --output-dir output \
    --retriever-model gpt-5-mini \
    --respond-model gpt-5 \
    --text-embedding-model Qwen/Qwen3-Embedding-4B \
    --gpu-list 0,1 \
    --num-workers 8
```

**Output**:
- `output/<retriever_model>_<respond_model>/egolife_eval_<PERSON>.json`
- `.log/egolife/eval/<PERSON>/eval_<PERSON>_*.log`

## Key Arguments

- `--person`: EgoLife subject ID, such as `A1_JAKE`
- `--base-dir` / `--data-dir`: EgoLife dataset directory
- `--output-dir`: Root directory for generated memory, semantic graph, cache inputs, and evaluation outputs
- `--model`: Model used during memory construction
- `--retriever-model`: Model used for retrieval-time selection and semantic graph file naming
- `--respond-model`: Model used to generate final answers
- `--text-embedding-model`: Text embedding model ID or local path for dense episodic and semantic retrieval
- `--gpu` / `--gpu-list`: GPU assignment for embedding precompute and parallel evaluation
- `--env-file`: Explicit environment file to load API credentials and model settings

**Note**: Relative `data/EgoLife` and `output` paths are interpreted from the current working directory. For open-source usage and cluster runs, prefer passing explicit paths when data or outputs live outside this repository.

<span id='results'/>

## 📊 Results

The following results are reported in the EM²Mem paper. Accuracy is reported in percent.

### EgoLifeQA

| Method | Ent. | EvR. | Hab. | Rel. | Task | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B | 35.2 | 30.2 | 39.3 | 46.4 | 46.0 | 38.6 |
| Gemini 2.5 Pro | 43.2 | 40.5 | 41.0 | 55.2 | 52.4 | 46.4 |
| GPT-5 | 47.2 | 42.1 | 47.5 | 53.6 | 55.6 | 48.6 |
| VideoChat-Flash | 28.8 | 32.5 | 37.7 | 37.6 | 38.1 | 34.2 |
| Time-R1 | 39.2 | 50.8 | 65.6 | 48.8 | 47.6 | 48.8 |
| Video-RTS | 40.8 | 48.4 | 62.3 | 48.8 | 47.6 | 48.2 |
| LightRAG | 40.8 | 48.4 | 67.2 | 50.4 | 44.4 | 48.8 |
| HippoRAG | 48.8 | 60.3 | 70.5 | 60.8 | 66.7 | 59.6 |
| Video-RAG | 49.6 | 56.3 | 67.2 | 55.2 | 54.0 | 55.4 |
| EgoRAG | 40.0 | 56.3 | 62.3 | 54.4 | 52.4 | 52.0 |
| Ego-R1 | 51.2 | 53.2 | 63.9 | 50.4 | 50.8 | 53.0 |
| HippoMM | 45.6 | 53.2 | 70.5 | 55.2 | 58.7 | 54.6 |
| M3-Agent | 44.4 | 54.8 | 62.3 | 56.8 | 54.0 | 53.5 |
| WorldMM | 62.4 | 64.3 | 75.4 | 62.4 | 71.4 | 65.6 |
| WorldMM† | 57.6 | 65.1 | 68.9 | 68.8 | 60.3 | 64.0 |
| **EM²Mem** | **60.8** | 61.1 | 63.9 | **72.8** | **74.6** | **66.0** |

† denotes reproduced WorldMM results under the same evaluation setting as EM²Mem.

### Video-MME (L)

| Method | ARES | AREC | ATTR | CNT | ISYN | OCR | ORES | OREC | SPER | SRES | TPER | TRES | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-VL-8B | 62.2 | 54.0 | 51.9 | 43.8 | 68.1 | 42.9 | 62.9 | 57.4 | 33.3 | 45.5 | 33.3 | 67.0 | 61.0 |
| Gemini 2.5 Pro | 56.9 | 47.6 | 66.7 | 41.7 | 71.8 | 57.1 | 53.3 | 40.7 | 0.0 | 72.7 | 66.7 | 48.4 | 55.7 |
| GPT-5 | 71.1 | 69.8 | 70.4 | 47.9 | 88.3 | 57.1 | 75.8 | 74.1 | 33.3 | 72.7 | 50.0 | 75.8 | 74.3 |
| VideoChat-Flash | 35.0 | 42.9 | 37.0 | 31.3 | 34.4 | 42.9 | 60.0 | 46.3 | 33.3 | 54.5 | 33.3 | 46.2 | 44.1 |
| Time-R1 | 20.6 | 28.6 | 25.9 | 35.4 | 31.9 | 35.7 | 53.3 | 48.2 | 33.3 | 36.4 | 50.0 | 44.0 | 37.6 |
| Video-RTS | 43.3 | 52.4 | 40.7 | 39.6 | 33.7 | 42.9 | 60.8 | 53.7 | 33.3 | 45.5 | 50.0 | 49.5 | 47.9 |
| LightRAG | 41.7 | 30.2 | 40.7 | 35.4 | 54.0 | 50.0 | 46.7 | 61.1 | 33.3 | 45.5 | 50.0 | 52.8 | 46.6 |
| HippoRAG | 45.6 | 47.6 | 40.7 | 37.5 | 52.2 | 42.9 | 52.9 | 64.8 | 66.7 | 54.5 | 50.0 | 70.3 | 52.1 |
| Video-RAG | 51.7 | 47.6 | 37.0 | 39.6 | 49.7 | 57.1 | 62.1 | 68.5 | 66.7 | 45.5 | 50.0 | 68.1 | 55.4 |
| EgoRAG | 31.1 | 55.6 | 33.3 | 22.9 | 41.1 | 28.6 | 44.6 | 48.2 | 33.3 | 54.5 | 66.7 | 48.4 | 41.1 |
| Ego-R1 | 37.2 | 52.4 | 40.7 | 35.4 | 38.0 | 35.7 | 42.1 | 51.9 | 66.7 | 63.6 | 50.0 | 52.8 | 42.7 |
| HippoMM | 41.1 | 42.9 | 55.6 | 35.4 | 38.7 | 35.7 | 37.9 | 53.7 | 33.3 | 54.5 | 50.0 | 47.3 | 41.6 |
| M3-Agent | 52.2 | 57.1 | 59.3 | 45.8 | 51.5 | 42.9 | 54.6 | 64.8 | 33.3 | 45.5 | 50.0 | 71.4 | 55.3 |
| WorldMM | **81.1** | 73.0 | 70.4 | 54.2 | **85.3** | 42.9 | 75.0 | 77.8 | 33.3 | 72.7 | **66.7** | **79.1** | 76.6 |
| WorldMM† | 73.3 | 68.3 | **77.8** | 60.4 | 80.2 | 50.0 | 72.4 | **77.8** | 33.3 | **90.9** | **66.7** | 71.1 | 73.1 |
| **EM²Mem** | 77.2 | **76.2** | **77.8** | **64.6** | 80.7 | **64.3** | **77.0** | **77.8** | **33.3** | 81.8 | 50.0 | **79.1** | **76.8** |

† denotes reproduced WorldMM results under the same evaluation setting as EM²Mem.

### Inference Efficiency

| Metric | WorldMM | EM²Mem | Gain |
| :--- | :---: | :---: | :---: |
| Avg. latency / query (s) | 459.00 | 98.21 | 4.67× |
| Wall-clock evaluation time (s) | 229,502 | 6,138 | 37.39× |
| Input tokens | 31.95M | 13.29M | 58.42%↓ |
| Output tokens | 10.08M | 1.99M | 80.29%↓ |
| Total tokens | 42.03M | 15.27M | 63.66%↓ |

EM²Mem shifts multimodal alignment and graph organization to offline memory construction, so inference reads from pre-built event-indexed memory cells instead of repeatedly aligning isolated fragments.
