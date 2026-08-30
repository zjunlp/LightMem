#!/usr/bin/env bash
# Prime the tiktoken cache for offline use.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_PYTHON="$REPO_ROOT/../envs/lightmem/bin/python"
if [ ! -x "$DEFAULT_PYTHON" ]; then
  DEFAULT_PYTHON="python3"
fi

MODEL_DIR="${LIGHTMEM_MODEL_DIR:-$REPO_ROOT/../model}"
CACHE="${TIKTOKEN_CACHE_DIR:-$MODEL_DIR/tiktoken_cache}"
PYTHON="${LIGHTMEM_PYTHON:-$DEFAULT_PYTHON}"

mkdir -p "$CACHE"

for name in o200k_base cl100k_base; do
  url="https://openaipublic.blob.core.windows.net/encodings/${name}.tiktoken"
  key=$("$PYTHON" -c "import hashlib,sys;print(hashlib.sha1(sys.argv[1].encode()).hexdigest())" "$url")
  dst="$CACHE/$key"

  if [ -s "$dst" ]; then
    echo "  $name: already cached"
    continue
  fi

  ok=0
  for attempt in 1 2 3 4 5; do
    if curl -sSL --retry 2 --max-time 300 -o "$dst.part" "$url" && [ -s "$dst.part" ]; then
      mv "$dst.part" "$dst"
      ok=1
      break
    fi
    sleep $((attempt * 3))
  done

  if [ "$ok" = 1 ]; then
    echo "  $name: downloaded ($(du -h "$dst" | cut -f1))"
  else
    rm -f "$dst.part"
    echo "  $name: FAILED" >&2
    exit 1
  fi
done

TIKTOKEN_CACHE_DIR="$CACHE" "$PYTHON" - <<'EOF'
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")
print(f"  verified: gpt-4o-mini -> {enc.name}")
EOF
