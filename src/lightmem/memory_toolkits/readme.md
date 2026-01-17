# Memory Baseline Framework

A comprehensive evaluation framework for benchmarking various memory layers on long-term conversational memory tasks. This framework provides a unified pipeline for **memory construction**, **memory retrieval**, and **question answering evaluation**.

---

## ✨ Key Features

- **Checkpoint, Recovery & Rerun**: Automatically saves progress during memory construction. If interrupted, simply re-run the script and it will skip already-processed trajectories and resume from where it left off. Use the `--rerun` flag to force rebuild memories from scratch when needed.
- **Parallel Processing**: Supports both **multi-threading** (within a single process) and **multi-process** execution (via `run_baseline.sh`). You can configure different `base_url` and `api_key` pairs for each process to distribute API load and accelerate evaluation.
- **Flexible Batch Runs**: Run on subsets of data by specifying `ranges` in `run_baseline.sh` or using the `--sample-size` argument for quick testing and debugging.
- **Non-Invasive Token Cost Monitoring**: Built-in token consumption tracking for LLM API calls. Uses monkey-patching to intercept calls **without modifying any baseline's internal code**.
- **Modular Architecture**: Clean separation between memory layers, datasets, and evaluation logic.
- **Multiple Baselines**: Supports A-MEM, LangMem, MemZero, MemZeroGraph, FullContext, and NaiveRAG.
- **Multiple Datasets**: Compatible with LongMemEval and LoCoMo benchmarks.

---

## 📁 Project Structure

```
memory_toolkits/
├── memory_construction.py    # Stage 1: Build memories from trajectories
├── memory_search.py          # Stage 2: Retrieve memories for each query
├── memory_evaluation.py      # Stage 3: Answer questions and evaluate
├── run_baseline.sh           # Bash script for parallel execution
├── configs/                  # Example configuration files
│   ├── A-MEM.json
│   ├── LangMem.json
│   ├── MemZero.json
│   ├── MemZeroGraph.json
│   ├── FullContext.json
│   ├── NaiveRAG.json
│   └── api_eval.json
├── envs/                     # Requirements for each baseline
│   ├── amem_requirements.txt
│   ├── langmem_requirements.txt
│   ├── mem0_requirements.txt
│   ├── fullcontext_requirements.txt
│   └── rag_requirements.txt
├── memories/
│   ├── datasets/             # Dataset loaders
│   │   ├── longmemeval.py
│   │   └── locomo.py
│   └── layers/               # Memory layer implementations
│       ├── amem.py
│       ├── langmem.py
│       ├── memzero.py
│       ├── full_context.py
│       └── naive_rag.py
├── inference_utils/          # QA and evaluation operators
├── token_monitor.py          # Token consumption tracking
└── monkey_patch.py           # Utility for patching LLM calls
```

---

## 🛠️ Installation

### Prerequisites

- **Python >= 3.12** is required.
- **Conda** (Anaconda or Miniconda) is recommended for environment management.

### Setting Up the Environment

> ⚠️ **Important**: Different memory baselines may have **conflicting dependencies**. We strongly recommend creating a **separate virtual environment for each baseline** to avoid dependency conflicts.

Each memory baseline has its own requirements file in the `envs/` directory.

**Example: Setting up environment for A-MEM**

```bash
# Create a new conda environment for A-MEM
conda create -n amem_env python=3.12 -y
conda activate amem_env

# Install dependencies
pip install -r envs/amem_requirements.txt
```

**Example: Setting up environment for LangMem**

```bash
# Create a separate conda environment for LangMem
conda create -n langmem_env python=3.12 -y
conda activate langmem_env

# Install dependencies
pip install -r envs/langmem_requirements.txt
```

**Example: Setting up environment for MemZero / MemZeroGraph**

```bash
# Create a separate conda environment for MemZero
conda create -n memzero_env python=3.12 -y
conda activate memzero_env

# Install dependencies
pip install -r envs/mem0_requirements.txt
```

**Other baselines:**

```bash
# For FullContext:
conda create -n fullcontext_env python=3.12 -y && conda activate fullcontext_env
pip install -r envs/fullcontext_requirements.txt

# For NaiveRAG:
conda create -n rag_env python=3.12 -y && conda activate rag_env
pip install -r envs/rag_requirements.txt
```

---

## 📖 Evaluation Pipeline Overview

The evaluation of all memory baselines follows a **three-stage pipeline**:

### Stage 1: Memory Construction

In this stage, user interaction trajectories are fed **incrementally** (message by message) into the memory layer. The memory layer builds and updates its internal memory state as each message arrives. This simulates real-world scenarios where memories are constructed over time.

- Some memory layers (e.g., **A-MEM**) perform **online operations** during construction, such as periodically updating embeddings after a certain number of memory additions. These operations are **forced to execute** after the entire trajectory has been processed.
- **Parallel processing** is applied at the trajectory level to accelerate evaluation, as memory construction is the most time-consuming stage.

### Stage 2: Memory Retrieval

Given the constructed memory, this stage retrieves the top-k most relevant memory units for each evaluation query. The retrieval results are saved to a JSON file for the next stage.

### Stage 3: Question Answering & Evaluation

Using the retrieved memories as context, a QA model generates answers for each question. A judge model then evaluates whether the generated answers match the ground truth, producing final accuracy metrics.

---

## 🚀 Quick Start: LongMemEval Example

This section walks you through running a complete evaluation on the **LongMemEval** dataset.

### Step 1: Download the Dataset

Download the LongMemEval dataset from HuggingFace:

👉 [https://huggingface.co/datasets/xiaowu0162/longmemeval](https://huggingface.co/datasets/xiaowu0162/longmemeval)

Save it to a local path, e.g., `/path/to/longmemeval.json`.

### Step 2: Create a Configuration File

Each memory layer requires a configuration file. Example configurations are available in the `configs/` directory. You can also check the full list of configuration fields in `memories/layers/` (e.g., `memories/layers/amem.py` for A-MEM).

**Example: A-MEM Configuration** (`my_amem_config.json`)

```json
{
  "user_id": "dummy",
  "embedder_provider": "openai",
  "retriever_name_or_path": "text-embedding-3-small",
  "base_url": "https://your-api-endpoint.com/v1",
  "llm_backend": "openai",
  "llm_model": "gpt-4o-mini",
  "evo_threshold": 100,
  "api_key": ""
}
```

> **Note**: The `user_id` field is a placeholder that will be overwritten during execution. Leave `api_key` empty if you prefer setting it via environment variables.

**Example: LangMem Configuration** (`my_langmem_config.json`)

```json
{
  "user_id": "dummy",
  "retriever_name_or_path": "openai:text-embedding-3-small",
  "retriever_dim": 1536,
  "llm_model": "openai:gpt-4o-mini",
  "query_limit": 40
}
```

**Example: MemZero Configuration** (`my_memzero_config.json`)

```json
{
  "user_id": "dummy",
  "llm_backend": "openai",
  "llm_model": "gpt-4o-mini",
  "embedder_provider": "openai",
  "retriever_name_or_path": "text-embedding-3-small",
  "embedding_model_dims": 1536,
  "use_gpu": "cuda"
}
```

### Step 3: Configure and Run the Baseline Script

Edit `run_baseline.sh` with your settings:

```bash
# ========================================================
# Configuration Section
# ========================================================
memory_type="A-MEM"                       # Options: A-MEM, LangMem, MemZero, MemZeroGraph, FullContext, NaiveRAG
dataset_type="LongMemEval"                # Options: LongMemEval, LoCoMo
dataset_path="/path/to/longmemeval.json"  # Path to your dataset file
config_path="/path/to/my_amem_config.json" # Path to your config file
num_workers=4                             # Number of parallel threads (avoid too many to prevent API rate limits)
tokenizer_path="gpt-4"                    # Tokenizer for token counting
log_dir="amem_logs"                       # Directory for log files
token_cost_prefix="token_cost"            # Prefix for token cost files
pid_prefix="process"                      # Prefix for process ID files

# Data ranges to process (start_idx end_idx)
# Each range runs as a separate process with its own API key
ranges=(
    "0 100"
    "100 200"
    "200 300"
    "300 400"
    "400 500"
)

# API keys and base URLs for each range
# You can use different API keys to distribute load
api_keys=(
    "sk-your-api-key-1"
    "sk-your-api-key-2"
    "sk-your-api-key-3"
    "sk-your-api-key-4"
    "sk-your-api-key-5"
)

base_urls=(
    "https://api.openai.com/v1"
    "https://api.openai.com/v1"
    "https://api.openai.com/v1"
    "https://api.openai.com/v1"
    "https://api.openai.com/v1"
)
# ========================================================
```

**Run Stage 1 (Memory Construction)**:

```bash
bash run_baseline.sh
```

This will launch parallel processes for each data range. Wait until all processes finish. You can monitor progress in the log files under `amem_logs/`.

### Step 4: Run Memory Retrieval

After memory construction completes, run the retrieval stage:

```bash
python memory_search.py \
    --memory-type A-MEM \
    --dataset-type LongMemEval \
    --dataset-path /path/to/longmemeval.json \
    --config-path /path/to/my_amem_config.json \
    --num-workers 4 \
    --top-k 10 \
    --start-idx 0 \
    --end-idx 500
```

**Arguments**:

| Argument | Description |
|----------|-------------|
| `--memory-type` | Memory layer type (`A-MEM`, `LangMem`, `MemZero`, `MemZeroGraph`, `FullContext`, `NaiveRAG`) |
| `--dataset-type` | Dataset type (`LongMemEval`, `LoCoMo`) |
| `--dataset-path` | Path to the dataset file |
| `--config-path` | Path to the memory configuration JSON file |
| `--num-workers` | Number of parallel threads |
| `--top-k` | Number of memories to retrieve per query (default: 10) |
| `--start-idx` | Starting index of trajectories to process |
| `--end-idx` | Ending index of trajectories to process |
| `--strict` | (Optional) Raise error if no memory found for a user |

The output will be saved as `{memory_type}_{llm_model}_{dataset_type}_{top_k}_{start_idx}_{end_idx}.json`.

### Step 5: Run Evaluation

Finally, evaluate the retrieval results.

> 💡 **Local Model Support**: If you want to use local models for LLM-as-a-Judge (instead of API-based models), please install [vLLM](https://github.com/vllm-project/vllm) first:
> ```bash
> pip install vllm
> ```

```bash
python memory_evaluation.py \
    --search-results-path A-MEM_gpt-4o-mini_LongMemEval_10_0_500.json \
    --qa-model gpt-4o-mini \
    --judge-model gpt-4o-mini \
    --qa-batch-size 4 \
    --judge-batch-size 4 \
    --dataset-type LongMemEval \
    --api-config-path /path/to/api_config.json
```

**Arguments**:

| Argument | Description |
|----------|-------------|
| `--search-results-path` | Path to the retrieval results JSON file |
| `--qa-model` | Model for generating answers (default: `gpt-4o-mini`) |
| `--judge-model` | Model for judging answer correctness (default: `gpt-4o-mini`) |
| `--qa-batch-size` | Batch size for QA generation (default: 4) |
| `--judge-batch-size` | Batch size for judgment (default: 4) |
| `--dataset-type` | Dataset type (`LongMemEval`, `LoCoMo`) |
| `--api-config-path` | (Optional) Path to API config file with keys and base URLs |

**API Config File Format** (`api_congi.json`):

```json
{
    "api_keys": [
        "sk-your-api-key-1",
        "sk-your-api-key-2"
    ],
    "base_urls": [
        "https://api.openai.com/v1",
        "https://api.openai.com/v1"
    ]
}
```

Alternatively, you can set environment variables instead of using `--api-config-path`:

```bash
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"
```

The evaluation results will be saved as `{search_results_path}_evaluation.json`.

---

## 📋 Memory Construction Arguments Reference

| Argument | Description |
|----------|-------------|
| `--memory-type` | Memory layer type (required) |
| `--dataset-type` | Dataset type (required) |
| `--dataset-path` | Path to the dataset file (required) |
| `--config-path` | Path to memory configuration JSON file |
| `--num-workers` | Number of parallel threads (default: 4) |
| `--start-idx` | Starting index of trajectories |
| `--end-idx` | Ending index of trajectories |
| `--rerun` | Force rebuild memories from scratch |
| `--sample-size` | Randomly sample a subset of the dataset |
| `--seed` | Random seed for sampling (default: 42) |
| `--token-cost-save-filename` | Filename for token cost statistics |
| `--tokenizer-path` | Path/name of tokenizer for token counting |

---

## 🔧 Supported Memory Layers

| Memory Layer | Description | Config Example |
|--------------|-------------|----------------|
| **A-MEM** | Agentic memory with evolution mechanism | `configs/A-MEM.json` |
| **LangMem** | LangChain-based memory with semantic search | `configs/LangMem.json` |
| **MemZero** | Mem0 memory system | `configs/MemZero.json` |
| **MemZeroGraph** | Mem0 with graph-based relations | `configs/MemZeroGraph.json` |
| **FullContext** | No memory layer, uses full conversation context | `configs/FullContext.json` |
| **NaiveRAG** | Simple RAG-based retrieval | `configs/NaiveRAG.json` |

---

## 📊 Supported Datasets

| Dataset | Description | Download Link |
|---------|-------------|---------------|
| **LongMemEval** | Long-term memory evaluation benchmark | [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval) |
| **LoCoMo** | Long-context conversational memory benchmark | [GitHub](https://github.com/snap-research/locomo/tree/main/data) |

---

## 💡 Tips

1. **API Rate Limits**: Set `num_workers` conservatively (e.g., 4-8) to avoid upstream API overload.
2. **Resume Interrupted Runs**: If the process is interrupted or encounters errors, simply re-run the same command. Completed trajectories will be skipped automatically and the process will resume from where it left off.
3. **Token Cost Tracking**: Check the generated `token_cost_*.json` files for detailed token consumption statistics.
4. **Log Files**: Monitor `{log_dir}/process_*.log` files for real-time progress and debugging.

---

## 📝 License

Please refer to the main project license.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. Ohter baselines will be added 🤗.
