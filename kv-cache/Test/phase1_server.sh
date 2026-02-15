#!/usr/bin/env bash
# Launch vLLM HTTP server with automatic prefix caching enabled for OPT-125m.
# Usage: bash phase1_server.sh

set -euo pipefail

# Avoid triton cache issues with spaces in paths
export TRITON_CACHE_DIR=/tmp/triton_cache

CHAT_TEMPLATE='{% for message in messages %}{{ message["content"] }}{% endfor %}'

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.7 \
    --max-model-len 1024 \
    --enforce-eager \
    --swap-space 0 \
    --port 8000 \
    --chat-template "$CHAT_TEMPLATE"
