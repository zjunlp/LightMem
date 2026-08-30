# LightMem Web Console

Web interface for configuring LightMem, writing memories, inspecting the
pipeline, and retrieving stored memories.

## Structure

```text
web/
├── backend/          FastAPI API and LightMemory process owner
│   └── app/
├── frontend/         React, TypeScript, Vite, and Tailwind CSS
└── scripts/          Runtime setup utilities
```

Runtime files are stored under `web/data/` and are excluded from Git.

## Run

Build the frontend, then start the backend:

```bash
cd web/frontend
npm install
npm run build

cd ../..
web/backend/run.sh
```

Open <http://127.0.0.1:8077>. The FastAPI process serves both the API and the
built frontend.

The backend must use a single worker because LightMemory keeps process-level
state and the embedded Qdrant store locks its data directory.

## Development

Run the backend and Vite separately:

```bash
web/backend/run.sh --reload

cd web/frontend
npm run dev
```

Vite runs on <http://127.0.0.1:5177> and proxies `/api` to port `8077`.

## Configuration

| Variable                   | Default                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| `LIGHTMEM_PYTHON`          | `../envs/lightmem/bin/python`, then `python3`                      |
| `LIGHTMEM_WEB_HOST`        | `127.0.0.1`                                                        |
| `LIGHTMEM_WEB_PORT`        | `8077`                                                             |
| `LIGHTMEM_WEB_DATA`        | `web/data`                                                         |
| `LIGHTMEM_MODEL_DIR`       | `<repository parent>/model`                                        |
| `LIGHTMEM_MODEL_ROOTS`     | model directory and Hugging Face cache                             |
| `LIGHTMEM_COMPRESSOR_PATH` | `<model dir>/llmlingua-2-bert-base-multilingual-cased-meetingbank` |
| `LIGHTMEM_EMBEDDER_PATH`   | `<model dir>/all-MiniLM-L6-v2`                                     |
| `TIKTOKEN_CACHE_DIR`       | `<model dir>/tiktoken_cache`                                       |

API credentials are stored in `web/data/secrets.json` with mode `0600`. The
browser only receives a masked key.

## Tiktoken Cache

Prime the tokenizer cache if the runtime cannot reach the tiktoken CDN:

```bash
web/scripts/prime_tiktoken_cache.sh
```

The script uses the same `LIGHTMEM_PYTHON` and `TIKTOKEN_CACHE_DIR` settings as
the backend.

## Implementation Notes

- Mutating operations run through a serial job queue.
- Job progress is streamed to the browser with server-sent events.
- The pipeline view parses LightMem debug logs, so the provided presets use
  `DEBUG` logging.
- Retrieval calls the configured retriever directly to preserve scores and
  payload metadata.
- `online_update()` is not implemented in the current LightMem release; the
  console uses the offline update path.
