# StructMem Multimodal

The multimodal StructMem implementation is a standalone service for storing and retrieving ordered text and image evidence.
It is an active research implementation for multimodal evaluation and is maintained independently from the text project.

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

## ✨ Capabilities

- Persistent SQLite storage with user isolation.
- Original text and image evidence returned with search results.
- Remote multimodal embeddings, token-level late interaction, and lexical retrieval.
- Optional local CLIP encoding and reranking integrations.
- FastAPI endpoints with explicit limits and configuration semantics.

## 🚀 Install and run

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export MEMORY_API_KEY='change-me'
python -m uvicorn service.multimodal:app --host 127.0.0.1 --port 18084
```

Open `http://127.0.0.1:18084/docs` for the interactive API schema.
Remote embedding providers and optional local models are configured through the variables listed in [`.env.example`](.env.example); the service reads process environment variables and does not load the file automatically.

## 🔎 Retrieval and model configuration

Search ranks the requesting user's stored source records and returns source content, optionally with explicitly marked same-session context.
The service does not generate an answer or use an LLM abstention gate in its Search path.
Query vectors may be cached temporarily in process memory; Search queries are not added to the persistent memory database by this service.
The example configuration names `text-embedding-v4` and uses text-only embeddings; a configured compatible endpoint and key are required to activate it.
Image-only queries require an image-capable encoder, and image-only stored records cannot be scored by a text-only encoder without supplied text.
Optional local CLIP, joint-embedding and reranker backends have separate model requirements and should be disclosed when used.
For evaluations with model restrictions, configure only permitted models and record the actual running configuration; the example is not evidence of any historical deployment.

## 🔌 API

- `GET /health`: report configuration and encoder readiness.
- `POST /v1/memories/add`: persist messages and original content.
- `POST /v1/memories/search`: retrieve evidence for a user.

Use `Authorization: Bearer <key>` or `X-API-Key: <key>`.
See [`SERVICE.md`](SERVICE.md) for request limits, storage semantics, and tuning options.

## 🧪 Tests

```bash
python -m pytest -q
```

The tests use synthetic content and mocked remote embeddings.
They require no external data, model downloads, or provider credentials.

## Attribution and license

This directory is the multimodal StructMem implementation.
It does not claim a separate published paper.
Released under the MIT License; see [`LICENSE`](LICENSE).
