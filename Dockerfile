ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=300

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEMORY_DATA_DIR=/data

WORKDIR /app

COPY pyproject.toml README.md LICENSE requirements-service.txt ./
COPY src ./src
COPY service ./service

RUN pip install --no-cache-dir --retries 10 \
      --default-timeout "${PIP_DEFAULT_TIMEOUT}" \
      --index-url "${PIP_INDEX_URL}" \
      -r requirements-service.txt \
    && pip install --no-cache-dir --no-deps -e . \
    && mkdir -p /data \
    && useradd --create-home --uid 65532 appuser \
    && chown -R 65532:65532 /app /data

USER 65532:65532
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
