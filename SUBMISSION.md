# StructMem Raw Hybrid Agent Memory Leaderboard Submission

## API

The service implements the AML Add/Search contract:

- `GET /health`
- `POST /add`
- `POST /search`

Add accepts `request_id`, `messages`, `user_id`, and `session_id`. It persists
every message before returning HTTP 200. Exact retries are idempotent, while a
reused `request_id` with a different payload returns HTTP 409.

Search accepts `query`, `user_id`, `top_k`, and optional `options`. It returns a
top-level `data` array containing no more than `top_k` results, scoped to the
exact `user_id`. Empty users return immediately without an embedding request.

## Build and Run

The default one-command installation builds the CPU-only image, downloads the
pinned BGE weights into it, and starts the service:

```bash
docker compose up --build -d
```

If Docker Hub or Hugging Face is not directly reachable during build, the image
source, CPU PyTorch wheel, Python package index, and Hugging Face endpoint can
all be overridden without editing repository files. For example:

```bash
PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim \
PYTORCH_WHEEL_URL=https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.5.1%2Bcpu-cp311-cp311-linux_x86_64.whl \
HF_ENDPOINT=https://hf-mirror.com \
docker compose up --build -d
```

The equivalent manual commands are:

```bash
docker build -t structmem-raw-hybrid:0.3.0 .
docker run --rm -p 8000:8000 \
  -v structmem-data:/data \
  structmem-raw-hybrid:0.3.0
```

The image installs the CPU-only PyTorch wheel and downloads
`BAAI/bge-small-en-v1.5` during `docker build`. The Hugging Face revision is
pinned to `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, and the resulting local
model is stored at `/opt/models/bge-small-en-v1.5`. After the image has been
built, startup and inference are fully offline and do not require CUDA, an
embedding API, or a model volume.

Set `MEMORY_EMBEDDING_PROVIDER=openai` and provide an API key to use
`text-embedding-3-small` through an OpenAI-compatible endpoint instead. The
provider, model, revision, query prefix, and document format form the persisted
embedding identity; startup fails if a database is opened with an incompatible
configuration. `MEMORY_EMBEDDING_MODEL` remains a legacy common override; the
unambiguous variables are `MEMORY_LOCAL_EMBEDDING_MODEL` and
`MEMORY_OPENAI_EMBEDDING_MODEL`.

## Method

Version 0.3 is a lossless evidence-retrieval ablation of the previous StructMem
submission:

1. Add stores each raw message, original role, session, and source timestamp;
   only the unmodified message content is embedded.
2. Search embeds the question and optional candidate answers, and computes
   cosine similarity against normalized message vectors.
3. A BM25 lexical ranking is fused with the dense ranking using Reciprocal Rank
   Fusion (`k=60`, lexical weight `0.5`).
4. The first 20 retrieval seeds expand to one adjacent message on either side
   within the same session.
5. For the first 10 seeds, one cross-session semantic neighbor is added when
   cosine similarity is at least `0.72`. This reuses stored vectors and adds no
   model call, generated memory, or graph database. Results are deduplicated and
   capped at `top_k`.
6. Returned content is original evidence with role and source time; the memory
   service does not generate answers.

The AML container intentionally does not run the repository's LoCoMo-specific
factual/relational extraction prompts or cross-event summary module. Those
research components remain in the source repository for the original StructMem
method, but were removed from this leaderboard path because generated rewrites
lost exact evidence and summaries competed with event results for the response
budget.

Runtime tuning variables are `MEMORY_LEXICAL_WEIGHT`, `MEMORY_RRF_K`,
`MEMORY_RESULT_WINDOW`, `MEMORY_RESULT_WINDOW_SEED_K`,
`MEMORY_SEMANTIC_BRIDGE_SEED_K`, and
`MEMORY_SEMANTIC_BRIDGE_MIN_SIMILARITY`. Set the semantic seed count to `0` for
the raw-hybrid baseline.

## Validation

The repository includes contract, persistence, user-isolation, result-cap, and
semantic-bridge tests in `tests/test_aml_contract.py`. The frozen v0.3 settings
were also replayed on LoCoMo and on a fixed, stratified 10% LongMemEval sample;
the detailed retrieval-proxy results and sampling seed are recorded in
`AML_OPTIMIZATION.md`. These proxy measurements are not official AML scores.

## Attribution

The original StructMem method is by Buqiang Xu, Yijun Chen, Jizhan Fang, Ruobin
Zhong, Yunzhi Yao, Yuqi Zhu, Lun Du, and Shumin Deng.

Technical report: https://arxiv.org/abs/2604.21748

The v0.3 ablation was informed by the public AML implementations ReFind and
vanilla-rag-memory/InvMem, particularly their preservation of raw evidence,
lexical retrieval, rank fusion, option-aware search, and local context windows:

- https://github.com/imlrz/ReFind
- https://github.com/wenxiaof345-ctrl/vanilla-rag-memory

The adapter, persistence layer, retrieval implementation, and tests in this
repository are maintained here; no source files are copied from those projects.
