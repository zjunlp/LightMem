# StructMem Multimodal Service

Start with `python -m uvicorn service.multimodal:app --host 127.0.0.1 --port 18084`.
This entry point uses its own database and does not import the text service.
The API runtime needs FastAPI, Uvicorn, Pydantic, NumPy, HTTPX and Pillow.
Local CLIP additionally needs PyTorch and Transformers.

## Configuration

Keep credentials in a private environment file outside the repository.

| Variable | Purpose |
| --- | --- |
| `MEMORY_API_KEY` | Required incoming API authentication |
| `MULTIMODAL_DB` | Independent SQLite database path |
| `MULTIMODAL_EMBEDDING_BASE_URL` | Embedding provider URL ending in `/v1` |
| `MULTIMODAL_EMBEDDING_API_KEY` | Provider authentication |
| `MULTIMODAL_EMBEDDING_MODEL` | Provider model identifier; defaults to `text-embedding-v4` |
| `MULTIMODAL_EMBEDDING_INPUT_MODE` | `text_only` or `multimodal_parts`; defaults to `text_only` |
| `MULTIMODAL_EMBEDDING_DIMENSIONS` | Requested vector size; defaults to 1024 |
| `MULTIMODAL_EMBEDDING_FORMAT` | `singlevector` or `multivector`; defaults to `singlevector` |
| `MULTIMODAL_EMBEDDING_CONCURRENCY` | Global simultaneous embedding requests; default 4 |
| `MULTIMODAL_EMBEDDING_BATCH_SIZE` | Maximum parts per provider request |
| `MULTIMODAL_EMBEDDING_IMAGE_BATCH_SIZE` | Maximum images per provider request |
| `MULTIMODAL_EMBEDDING_TIMEOUT` | Timeout in seconds for each provider attempt |
| `MULTIMODAL_SCORING` | `late`, `modality_zmax`, or `modality_rrf` |
| `MULTIMODAL_LEXICAL_WEIGHT` | BM25 fusion weight; set 0 for dense ranking alone |

The default open-source profile uses the text-only `text-embedding-v4` endpoint.
Set the provider URL and key to enable it.
In `text_only` mode, the service sends only text parts to the embedding endpoint.
Image parts remain stored and are returned unchanged, but do not receive dense vectors.
Text queries retrieve records with text vectors; image-only queries return 503 because this model cannot embed images.
Set `MULTIMODAL_EMBEDDING_INPUT_MODE=multimodal_parts` to use a provider that accepts both text and inline image inputs.
For token and image patch embeddings from a multimodal provider, choose `multivector`, `late` scoring, lexical weight 0, and small batches such as 4 parts.
Every returned token vector is retained and normalized.
Query scoring averages the best document match for each query vector; vectors are not pooled into a single message vector.
Optional modality calibration compares text and image channels separately.
Image-containing queries retain direct late interaction.

Optional embedding input preprocessing is controlled by `MULTIMODAL_EMBEDDING_IMAGE_THRESHOLD_BYTES` (default 0, disabled), `MULTIMODAL_EMBEDDING_IMAGE_MAX_EDGE` (default 2048), and `MULTIMODAL_EMBEDDING_IMAGE_JPEG_QUALITY` (default 95).
Only images exceeding the decoded byte threshold are transformed, using EXIF orientation correction, RGB conversion, aspect-preserving downscaling and JPEG encoding.
Original evidence is stored and returned unchanged.
Provider, model, response format and enabled preprocessing form the stored encoder identity.
An API model identifier alone does not pin the provider's underlying weights.

## Contract and storage

- `GET /health` reports configuration, encoder initialization and readiness.
  A configured model is not ready until the first successful encoding.
- `POST /v1/memories/add` and `POST /v1/memories/search` require authentication.
  `/add` and `/search` are aliases.
- Content uses ordered `text` and `image_url` parts.
  Images are inline Base64 JPEG, PNG or WebP data URIs.
  Remote image URLs are rejected.
- Add encodes synchronously, then commits all records atomically.
  An exact retry is idempotent; conflicting reuse of a request identifier returns 409.
- Source role, session and timestamp are persisted.
  Search is scoped to the exact user, across that user's sessions, and returns at most `top_k` records.
- Vector matrices preserve part boundaries and modality metadata.
  A changed encoder identity causes old records to be encoded before dense use.
- Limits are 10 MiB decoded per image, 30 MiB images per Add or Search output, and 42 MiB request wire size.
  Search retains the largest complete ranked prefix that fits the image budget, stopping at the first record that would exceed it.
  Records are not skipped, split or rewritten to fit, so fewer than `top_k` records may be returned.
  If the first record alone exceeds the budget, Search returns 413.
  An exact budget match is accepted.
- Encoding failures return 503 without committing a partial Add.
  API calls have bounded parallelism, bounded pending work and finite retries.
- Search returns evidence only.
  It does not generate answers or consult benchmark labels, reference answers or dataset-specific identifiers.

Query vectors are cached in process for a default 300 seconds, using exact user, encoder identity and ordered content as the key.
Concurrent duplicate queries share one encoding operation.
This cache stores no Search results: newly added memories participate immediately.
Configure `MULTIMODAL_QUERY_CACHE_MB` (default 64), `MULTIMODAL_QUERY_CACHE_MAX_ENTRIES` (default 128), and `MULTIMODAL_QUERY_CACHE_TTL_SECONDS` (default 300); a zero limit disables it.
Entries expire or leave the cache under LRU and memory limits, and the cache is cleared at shutdown.
A provider may return different image vectors for repeated requests, so caching stabilizes exact query retries during its TTL.

Optional context scoring is disabled by default.
Set `MULTIMODAL_CONTEXT_WEIGHT` to a value in `[0, 1]` to blend each dense score with the maximum of itself and its immediate storage neighbors.
Only records in the same user's nonempty session with consecutive storage sequence numbers can share scores.
Updates use the original scores for one hop; evidence is returned unchanged.
This option requires an encoder, `late` scoring and zero lexical weight.
Validate its retrieval tradeoffs before enabling it.

Optional update evidence is disabled by default.
Set `MULTIMODAL_UPDATE_EVIDENCE=1` to supplement a selected source record with at most twelve missing turns from a short, already-selected same-session interval.
It activates only for queries that explicitly ask for a later/current/updated state (including Chinese markers such as `最后`, `后来`, `当前`, `调整`, or `承诺`).
Search keeps its original ranking and never returns more than `top_k` records; supplemental turns attach to the latest selected session anchor and are included in the existing response media budget.

Optional next-turn evidence is disabled by default.
Set `MULTIMODAL_NEXT_TURN_EVIDENCE=1` to attach the immediate next source turn when it belongs to the same nonempty session and has the next storage sequence.
This preserves ranking and the `top_k` result count, skips turns already added by update evidence, and remains subject to the same response media budget.

Optional bounded context expansion is disabled by default.
Set `MULTIMODAL_ANSWER_CONTEXT_TOP_K=5` to attach immediate previous and next source turns to the first five selected hits.
Expansion stays within the same nonempty user session and consecutive storage sequence.
Each source record appears at most once in the response, even when hit windows overlap.
The search ranking and result count remain unchanged, and appended images count toward the normal response media budget.
Evaluate the context cost and task-level tradeoffs before enabling this option.

Deployment and external evaluation registration are separate operational steps.

For an embedding endpoint that explicitly supports `prompt_name`, configure `MULTIMODAL_EMBEDDING_QUERY_PROMPT_NAME` and `MULTIMODAL_EMBEDDING_PASSAGE_PROMPT_NAME` to distinguish search queries from stored passages.
Both are unset by default, and the request omits the field.
These settings participate in encoder identity and query-cache keys; changing them causes old stored vectors to be encoded again when needed.
Use only prompt names supported by the endpoint.
Add and lazy document encoding use the passage profile; Search uses the query profile, including across parallel batches.

Optional capture metadata is disabled by default and requires an encoder.
Set `MULTIMODAL_CAPTURE_GAZETTEER` to a local GeoNames settlement TSV file to derive supplemental text from original image EXIF capture dates and GPS.
The text states the nearest settlement, country and distance, with an explicit uncertainty statement.
It does not infer the actual capture city or use filename dates.
GeoNames data is available under CC BY 4.0: https://www.geonames.org/.

The service reads EXIF from the original inline bytes before embedding image preprocessing.
Missing or invalid capture date/GPS adds no text.
Supplemental vectors participate in document retrieval only; query input and returned original content remain unchanged.
The parser version and settlement-file hash participate in encoder identity.
Add and lazy reencoding use the same metadata pipeline, including alignment of extra vectors with their source message.

Optional event completion: set `MULTIMODAL_EVENT_COMPLETION=1` with capture metadata, late scoring, zero lexical/context weights, consensus mode and source diversity disabled.
This preserves the first twenty results, inserts up to five event-completion candidates, and fills the remaining positions from the original top hundred.
Direct neighbors require single-image evidence with explicit EXIF capture offsets and GPS within one hour and one hundred metres.
Unknown `-00:00` offsets are excluded.
Event completion alternates the original dense order with neighbors of its first twenty candidates; it performs no transitive expansion.
Parsed capture fields, including missing values, are cached in SQLite per user and source sequence.
The option is disabled by default; returned original content and the thirty MiB stop-at-first-overflow budget still apply.

Optional source diversity: set `MULTIMODAL_PAIR_DIVERSITY=1` to reorder the first 100 retrieval candidates before applying `top_k`.
When both an image user message and its immediately following assistant reply are in that set, the image message represents the pair at its first ranked position and the reply moves to the end of the candidate set.
Pairs require the same user, a nonempty session, and consecutive storage sequences.
Other messages retain their relative order; candidates after the first 100 keep their positions.
Returned scores remain the original relevance scores, so they may no longer be descending.
The image byte budget is applied after reordering and stops at the first item that would exceed it.
This option is disabled by default.

Optional local relevance reranking is also disabled by default.
Install the `reranker` extra and point `MULTIMODAL_RERANKER_MODEL` to an existing local Qwen3-VL-Reranker model directory.
No model is downloaded by the service.
`MULTIMODAL_RERANKER_DEVICE` defaults to `cuda:0`.
`MULTIMODAL_RERANKER_MODE=consensus` combines equal local ranks from retrieval and reranking; `direct` uses the model's relevance order.
Only the first twenty candidates are reordered, before `top_k` and the original image byte budget.
Candidates after position twenty keep their positions.
Scores returned to the caller remain the original retrieval scores and may not be descending.

Scoring uses the query and original source images/text.
An image user message may include its immediate assistant reply as scoring context only when both records have the same nonempty session and consecutive storage sequences.
Returned evidence is unchanged.
Concurrent inference is bounded to one request; busy, invalid, overlong, failed or expired reranking preserves the original retrieval order.
`MULTIMODAL_RERANKER_TIMEOUT` defaults to 30 seconds, checked between model operations; it cannot interrupt a running model forward pass.
Health responses expose reranker configuration, loading and last error state.
Validate retrieval quality for the intended workload before enabling this option.

`MULTIMODAL_RERANKER_MODE=image_locked` preserves each image-bearing candidate at its original position and reorders only text candidates within the first twenty by local relevance.
Candidates after twenty retain their order.
This mode requires the optional local reranker; errors retain the original ranking.

`MULTIMODAL_RERANKER_MODE=session_diverse` scores text candidates within the first hundred, placing different source sessions before repeated sessions when relevant candidates are available.
Image candidates keep their positions.
Empty or overlong text candidates also keep their positions.
Other errors or an exhausted overall reranking budget retain the original ranking.
Candidates after one hundred retain their order.
This optional mode uses the configured local reranker.

`MULTIMODAL_RERANKER_MODE=session_reserved` uses session diversity while reserving source sessions for fixed candidates already present in the first ten positions before selecting text candidates.
Fixed candidates retain their positions, and candidates after one hundred retain their order.
It uses the same local reranker and fallback behavior as `session_diverse`.

`MULTIMODAL_RERANKER_BACKEND=joint_embedding` selects an optional local Qwen joint embedding scorer instead of the default `relevance` backend.
Set `MULTIMODAL_RERANKER_MODEL` to a local embedding model directory containing its upstream `scripts/qwen3_vl_embedding.py`, install the `reranker` extra, and use `MULTIMODAL_RERANKER_MODE=consensus`.
The scorer combines its local ranks with retrieval ranks within the first twenty candidates.
It accepts one text query and one original image per candidate.
When capture metadata is configured, the same source metadata passage is encoded jointly with each image.
Mixed content, multiple images, invalid model inputs, busy inference, or an exhausted budget retain the original retrieval ranking.
Original evidence is preserved.
Model loading is local only; no model is downloaded automatically.
The embedding cache holds at most 4096 vectors per process.

`MULTIMODAL_RERANKER_MODE=protected40` scores the first forty candidates in batches of at most twenty.
It retains the first ten candidates from direct relevance ordering of the original first twenty, then fills the remainder from relevance ordering of all forty.
Candidates after forty retain their order.
The complete operation shares the configured reranking time budget; a failed batch or exhausted budget retains the original retrieval ranking.
This optional mode uses the local relevance backend and performs more scoring work than `direct`.

`MULTIMODAL_RERANKER_MODE=protected20_tail100` scores up to one hundred candidates in batches of at most twenty.
It preserves the first twenty results of `protected40`, then orders the remaining scored candidates by relevance.
Candidates after one hundred retain their order.
All batches share one time budget, with the original ranking retained on failure.
This optional mode requires more inference than `protected40`.

`MULTIMODAL_DENSE_SESSION_EXPANSION=1` optionally preserves the first twenty results and inserts up to five dense representatives from sessions absent from the initial hundred results.
It is disabled by default and requires an encoder, zero lexical and context weights, and `session_diverse` mode.
The normal original media budget still applies.
All candidates come from the requesting user's current storage snapshot.
The expansion selects a bounded hundred-result prefix.
Requests beyond one hundred continue through the remaining original order, skipping representatives already moved into that prefix.

`MULTIMODAL_SESSION_IMAGE_SUPPORT=1` optionally adds up to five dense image representatives for sessions present in the first twenty results but lacking an image in the first hundred.
It requires dense session expansion and is disabled by default.
It preserves the first twenty and the remaining original relative order after moved records are removed; the normal media budget applies.
Images come from the same user storage snapshot.
On support-selection failure, search uses the ranking from before this optional step.
