#!/usr/bin/env bash
# Launch llama-server with GPT-2 (125M) GGUF for Phase 3a experiments.
# Usage: bash phase3a_server.sh

LLAMA_SERVER="/home/rifatxia/Desktop/Gnosis/Gnosis-Learning/nsight-systems/llama.cpp/build/bin/llama-server"
MODEL="/home/rifatxia/Desktop/Gnosis/Gnosis-Learning/nsight-systems/llama.cpp/models/gpt2.gguf"

export LD_LIBRARY_PATH="/home/rifatxia/Desktop/Gnosis/Gnosis-Learning/nsight-systems/llama.cpp/build/bin:${LD_LIBRARY_PATH}"

exec "$LLAMA_SERVER" \
    --model "$MODEL" \
    --ctx-size 2048 \
    --parallel 1 \
    --n-gpu-layers -1 \
    --port 8000 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --chat-template "{% for message in messages %}{{ message.content }}{% endfor %}"
