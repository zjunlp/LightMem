# StructMem Raw Hybrid Agent Memory Leaderboard Submission

## API

The service implements the AML Add/Search contract:

- `GET /health`
- `POST /add`
- `POST /search`

Add accepts `request_id`, `messages`, `user_id`, and `session_id`. It persists
every message before returning HTTP 200. Exact retries are idempotent, while a
reused `request_id` with a different payload returns HTTP 409.
Image media accepts both a direct URL string and the OpenAI form
`{"image_url": {"url": "..."}}`; captions are indexed with the message text.

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
docker build -t structmem-raw-hybrid:0.3.2 .
docker run --rm -p 8000:8000 \
  -v structmem-bgem3-data:/data \
  structmem-raw-hybrid:0.3.2
```

The image installs the CPU-only PyTorch wheel and downloads `BAAI/bge-m3`
during `docker build`. The Hugging Face revision is pinned to
`5617a9f61b028005a4858fdac845db406aefb181`, and the resulting local model is
stored at `/opt/models/bge-m3`. After the image has been built, startup and
inference are fully offline and do not require CUDA, an embedding API, or a
model volume. The model change uses 1024-dimensional vectors, so an existing
memory database created with another embedding identity must not be reused.

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
   embedding text includes the message role, body, and supplied media captions. When
   `MEMORY_EMBED_TIMESTAMPS=true`, a normalized session-date prefix is also
   included for timestamped messages; this option is disabled by default.
   `MEMORY_EMBED_SPEAKERS=false` disables the role prefix. The stored and
   returned evidence remains unchanged.
   `MEMORY_MULTI_WINDOW=true` uses overlapping 384-token windows (64-token
   overlap) for long messages. Search takes the highest window similarity for
   each original message and still returns only original messages. This mode
   uses a tokenizer-backed local embedder, versions its stored vectors, and
   defaults to RRF score fusion; `MEMORY_FUSION` can override the fusion mode.
2. Search embeds the question and optional candidate answers, and computes
   cosine similarity against normalized message vectors.
3. Dense and BM25 scores use MinMax fusion by default, while multi-window
   indexing defaults to RRF fusion. English-dominant corpora use lexical
   weight `0.1`; CJK-bearing queries in corpora with at least 5% CJK-bearing
   messages use `0.5`, including captions attached to image messages.
4. Search returns the highest-ranked raw evidence messages within `top_k`.
   Optional session diversity promotes one representative per session when
   `MEMORY_SESSION_DIVERSITY=true`; it can improve session coverage while
   reducing exact-message recall at the same return budget.
5. Returned content is original evidence with role and source time; the memory
   service does not generate answers.

`MEMORY_SESSION_RECENCY_WEIGHT` optionally favors newer sessions using their
first Add order. It defaults to `0`; `0.2` is a moderate setting when complete
sessions are added in chronological order. It changes dense ranking before
lexical fusion and does not change the stored evidence.

`MEMORY_SESSION_REPEAT_PENALTY` optionally subtracts a soft penalty from
standardized fused scores for the second, third, and later hits within each
session's original ranking. It does not impose a per-session result cap.
`MEMORY_SESSION_ANCHOR_SEED_K` considers sessions appearing in the first
configured number of selected results. If such a session occupies two or more
slots and its earliest raw user message is absent, that message replaces the
session's last slot. The first hit, represented sessions, and result count are
preserved. Both settings default to `0` and do not modify stored evidence.

With the repeat penalty alone, returned scores are the adjusted standardized
fusion scores. With anchor selection enabled, all returned scores are final
reciprocal ranks `1 / (MEMORY_RRF_K + rank)`, not similarity scores.

The AML container intentionally does not run the repository's LoCoMo-specific
factual/relational extraction prompts or cross-event summary module. Those
research components remain in the source repository for the original StructMem
method, but were removed from this leaderboard path because generated rewrites
lost exact evidence and summaries competed with event results for the response
budget.

Runtime tuning variables are `MEMORY_LEXICAL_WEIGHT`, `MEMORY_EMBED_SPEAKERS`,
`MEMORY_MULTI_WINDOW`, `MEMORY_FUSION`, `MEMORY_RRF_K`,
`MEMORY_SESSION_RECENCY_WEIGHT`,
`MEMORY_SESSION_REPEAT_PENALTY`, `MEMORY_SESSION_ANCHOR_SEED_K`,
`MEMORY_RESULT_WINDOW`, `MEMORY_RESULT_WINDOW_SEED_K`,
`MEMORY_SEMANTIC_BRIDGE_SEED_K`, and
`MEMORY_SEMANTIC_BRIDGE_MIN_SIMILARITY`. Set the semantic seed count to `0` for
the raw-hybrid baseline.

## Idle-time index preparation

Add still waits for raw-message encoding and durable storage before returning.
An idle-time worker then prepares an immutable per-user vector matrix and BM25
statistics for both ordinary and CJK tokenization. Appended messages reuse the
previous snapshot's token counts; the full-corpus document frequencies and
length statistics are updated before the new snapshot is published.

Search never waits for this worker. It uses a snapshot only when the database's
indexed-message count and maximum sequence match; otherwise it immediately uses
the original full-corpus retrieval path, including newly added messages. A
snapshot is published only after rechecking that database state. Deletions
invalidate snapshots, and restart requires no background recovery before Search
can serve persisted messages. No model-generated summaries or query-driven
policy changes run in this worker.

`MEMORY_BACKGROUND_INDEX=false` disables this optimization. The worker waits for
all online Add/Search calls to finish and an idle grace period (default 0.25s),
then yields between documents if online work arrives. Its Linux thread has lower
scheduling priority. This is cooperative scheduling, not instantaneous preemption
of database reads or NumPy copies. It does not call the embedding model.

The cache has configurable estimated memory, user-count and inactivity limits:
`MEMORY_INDEX_CACHE_MB=512`, `MEMORY_INDEX_MAX_USERS=32`, and
`MEMORY_INDEX_TTL_SECONDS=300`. Oversized users and failed builds retain the
synchronous fallback. Cached and baseline retrieval use the same BM25 formula,
fusion, context policy and original evidence. Expiry clears cached copies but
does not delete the durable database; evaluation data still requires operational
deletion within the platform's retention deadline.

## Validation

For CPU deployments, `MEMORY_EMBEDDING_BATCH_SIZE` controls model batching and
`MEMORY_EMBEDDING_MAX_CONCURRENCY` bounds simultaneous local encoder calls
(defaults: 64 and 2). Set a smaller batch for long messages and limited RAM.
Compose limits OpenMP/MKL threads to 2 per inference call by default; tune these
with measured throughput rather than increasing workers and duplicating models.
These settings control execution resources and do not change retrieval policy.

The repository includes contract, persistence, user-isolation, result-cap, and
semantic-bridge tests in `tests/test_aml_contract.py`. Local retrieval-proxy
measurements informed the defaults. They are not official leaderboard scores;
private experiment data and results are kept outside this repository.

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
