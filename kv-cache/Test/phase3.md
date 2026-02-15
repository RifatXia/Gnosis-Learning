Then llama
# Phase 3: llama.cpp

Same experiment, now on llama.cpp. Covers both the HTTP server (`llama-server`) and the Python bindings (`llama-cpp-python`).

## Key Difference: How Prefix Caching Works

vLLM uses an explicit radix tree with block-level hashing. llama.cpp takes a simpler approach — it keeps a single KV cache and does **longest prefix match** against whatever is already in the cache from the previous request. There is no multi-tenant tree; it's essentially a single-slot cache that checks how many leading tokens match.

This means:
- Prefix reuse is **automatic** — no `--enable-prefix-caching` flag needed
- Only the **last request's** KV cache is available for matching (no cross-request sharing)
- The cache is invalidated by any prefix change, same as vLLM

For the server, `--cache-type-k f16 --cache-type-v f16` controls KV precision. Slot-based caching (`--slot-save-path`) can persist KV state across requests for multi-slot scenarios.

## Phase 3a: HTTP Server (`llama-server`)

### Server launch (replaces vLLM server)

```bash
llama-server \
    --model models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
    --ctx-size 32768 \
    --n-gpu-layers -1 \
    --port 8000 \
    --cache-type-k f16 \
    --cache-type-v f16
```

No prefix caching flag — it's on by default. `--n-gpu-layers -1` offloads all layers to GPU.

### API calls (mostly identical)

llama.cpp's server exposes an OpenAI-compatible `/v1/chat/completions` endpoint. The Phase 1 HTTP code works as-is, with one difference — the `stream_options` field may not be supported. Use the `/health` and `/slots` endpoints to verify cache state:

```python
# Check slot cache status
async with session.get(f"{base_url}/slots") as resp:
    slots = await resp.json()
    # slots[0]["n_past"] shows how many tokens are cached
```

### TTFT measurement

Streaming works the same way as Phase 1. The SSE format is OpenAI-compatible:

```python
payload = {
    "model": "local-model",
    "messages": messages,
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": True,
    # no stream_options — use timing only
}
```

### Warming the cache

Same approach — send a request with [A, B] context and `max_tokens: 1`. However, since llama.cpp only caches the last request per slot, the warmup and measurement must hit the **same slot**. With a single-slot server (default), this is automatic.

For multi-slot (`--parallel N`), pin to a specific slot via the `id_slot` field:

```python
payload = {
    ...
    "id_slot": 0,  # pin to slot 0
}
```

## Phase 3b: Python Bindings (`llama-cpp-python`)

### Engine initialization (replaces vLLM `LLM`)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    n_ctx=32768,
    n_gpu_layers=-1,
    verbose=False,
)
```

### Prompt construction

`llama-cpp-python` has a built-in `create_chat_completion()` that handles templates, but for direct control use the Jinja template the same way:

```python
# Option A: Let llama-cpp-python handle the template
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=1,
    temperature=0.0,
)

# Option B: Manual template (same as vLLM Phase 2)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = llm(prompt, max_tokens=1, temperature=0.0)
```

Option A is simpler. Option B gives identical prompt construction to Phases 1 and 2 for apples-to-apples comparison.

### TTFT measurement

Same `max_tokens=1` timing approach as vLLM Phase 2:

```python
start = time.perf_counter()
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=1,
    temperature=0.0,
)
ttft_ms = (time.perf_counter() - start) * 1000

prompt_tokens = response["usage"]["prompt_tokens"]
```

### Token counts

Available in the response `usage` dict:

```python
prompt_tokens = response["usage"]["prompt_tokens"]
output_tokens = response["usage"]["completion_tokens"]
```

### Cache behavior difference

With `llama-cpp-python`, the KV cache persists in the `Llama` object between calls. The prefix match happens automatically — if the new prompt shares a prefix with the previous call, those KV entries are reused. No explicit warmup step is strictly necessary if calls are sequential, but we keep it for consistency.

**Critical**: calling `llm.reset()` or creating a new `Llama` instance clears the cache entirely. Avoid this between warmup and measurement.

## What Stays the Same

- Experiment design (warmup → T2 cache hit → T3 cache miss)
- Synthetic context generation
- Trial structure and metrics
- All parameters

## What's Different (Summary)

| Aspect | vLLM | llama.cpp |
|--------|------|-----------|
| Model format | HuggingFace (safetensors) | GGUF (quantized) |
| Prefix caching | Explicit flag, radix tree | Automatic, single-slot LPM |
| Multi-request sharing | Yes (tree) | No (last request only) |
| Cache warmup | Required (populate tree) | Implicit (sequential calls) |
| Slot pinning | N/A | Needed with `--parallel N` |
| Quantization | Separate from caching | Built into GGUF format |

## Dependencies

Phase 3a: `aiohttp`, `numpy` (same as Phase 1)
Phase 3b: `llama-cpp-python`, `numpy` (optionally `transformers` for manual templates)

```bash
pip install llama-cpp-python numpy
# For GPU support:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

## Caveat: Apples-to-Apples

The model weights differ between vLLM (fp16/bf16 safetensors) and llama.cpp (quantized GGUF). This affects absolute TTFT numbers. The experiment's conclusion — that removing context A causes a TTFT regression — holds regardless of backend, but direct numerical comparison between vLLM and llama.cpp requires controlling for quantization (e.g., using Q8_0 GGUF which is close to fp16).