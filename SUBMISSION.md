# StructMem Agent Memory Leaderboard Submission

## API

The service implements the AML Add/Search contract:

- `GET /health`
- `POST /add`
- `POST /search`

Add accepts `request_id`, `messages`, `user_id`, and `session_id`. Search accepts
`query`, `user_id`, `top_k`, and the optional `options` field. Add is idempotent
for the same request payload and returns `409` when a request ID is reused with
a different payload. Search returns a top-level `data` array and scopes all
results to the exact `user_id` supplied by the caller.

## Build and Run

```bash
docker build -t structmem-lightmem:latest .
docker run --rm -p 8000:8000 \
  -e MEMORY_LLM_API_KEY="$MEMORY_LLM_API_KEY" \
  -e MEMORY_EMBEDDING_API_KEY="$MEMORY_EMBEDDING_API_KEY" \
  -e MEMORY_LLM_BASE_URL="${MEMORY_LLM_BASE_URL:-https://api.openai.com/v1}" \
  -e MEMORY_EMBEDDING_BASE_URL="${MEMORY_EMBEDDING_BASE_URL:-https://api.openai.com/v1}" \
  -v structmem-data:/data \
  structmem-lightmem:latest
```

The LLM and embedding credentials are runtime configuration and are not stored
in the repository. The service uses `gpt-5.4-mini` and
`text-embedding-3-small` by default, with local Qdrant persistence under `/data`.
After each Add, event memories are persisted immediately. Cross-event
consolidation starts automatically when `MEMORY_SUMMARY_BATCH_ENTRIES` pending
events have accumulated (default: 20). Summary is intentionally not triggered
by Search; a final partial batch remains available as event-level memories and
does not cause an extra Summary LLM call.

## Method and Attribution

StructMem performs dual-view factual/relational event extraction, temporal
anchoring, and cross-event semantic consolidation. The original method is by
Buqiang Xu, Yijun Chen, Jizhan Fang, Ruobin Zhong, Yunzhi Yao, Yuqi Zhu, Lun Du,
and Shumin Deng.

Technical report: https://arxiv.org/abs/2604.21748

The service-specific changes are the AML HTTP adapter, request idempotency
registry, user-scoped Qdrant payloads and filters, combined event/summary
Search results, and Docker packaging. The underlying StructMem extraction and
consolidation design is from the cited method.
