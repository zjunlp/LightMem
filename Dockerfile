ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=300
ARG PYTORCH_WHEEL_URL=https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp311-cp311-linux_x86_64.whl
ARG BGE_MODEL_ID=BAAI/bge-m3
ARG BGE_MODEL_REVISION=5617a9f61b028005a4858fdac845db406aefb181
ARG HF_ENDPOINT=https://huggingface.co

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEMORY_DATA_DIR=/data \
    HF_HOME=/data/huggingface \
    MEMORY_LOCAL_EMBEDDING_MODEL=/opt/models/bge-m3 \
    MEMORY_LOCAL_EMBEDDING_REVISION=5617a9f61b028005a4858fdac845db406aefb181

WORKDIR /app

COPY requirements-service.txt ./

RUN pip install --no-cache-dir --retries 10 \
      --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
      "${PYTORCH_WHEEL_URL}" \
    && pip install --no-cache-dir --retries 10 \
      --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
      --index-url "${PIP_INDEX_URL}" \
      -r requirements-service.txt \
    && HF_ENDPOINT="${HF_ENDPOINT}" python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${BGE_MODEL_ID}', revision='${BGE_MODEL_REVISION}', device='cpu').save('/opt/models/bge-m3')" \
    && rm -rf /root/.cache/huggingface \
    && mkdir -p /data \
    && useradd --create-home --uid 65532 appuser \
    && chown -R 65532:65532 /app /data /opt/models

COPY --chown=65532:65532 service ./service

USER 65532:65532

RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('/opt/models/bge-m3', device='cpu')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
