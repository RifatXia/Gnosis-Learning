# Experiment 1: Prefix Cache Invalidation Cost

## Overview

This experiment measures the KV cache recomputation cost when context is removed from a multi-turn conversation served by vLLM with automatic prefix caching (APC) enabled. Removing a prefix block (e.g., due to context poisoning or context exhaustion) invalidates the cached KV entries for all downstream tokens, forcing a full prefill recomputation.

## Experiment Design

We construct a multi-turn conversation with two context blocks A and B, then measure the time-to-first-token (TTFT) and throughput when querying with the full prefix versus a truncated prefix.

| Step | Context Sent | Prefix Cache Status | What We Measure |
|------|-------------|-------------------|-----------------|
| T1 | Warm cache with [A, B] | — | Setup (populate cache) |
| T2 | Query Q with [A, B] | **HIT** ✅ | Baseline TTFT and throughput |
| T3 | Query Q with [B] only (A removed) | **MISS** ❌ | Degraded TTFT and throughput |

**Expected result**: T3.TTFT >> T2.TTFT, because removing A changes the token prefix from position 0 onward, invalidating every cached block.

## Why the Cache Misses

vLLM's prefix cache keys each block by the cumulative token sequence from position 0 up to that block. When A is removed, B's tokens now start at position 0 instead of position `len(A)`. This means:

- Every block hash changes (different cumulative prefix).
- RoPE positional encodings differ (same tokens, different positions).
- Causal attention context differs (B no longer attends to A).

The entire KV cache for B must be recomputed from scratch during prefill.

## Technical Setup

### vLLM Server Configuration

Start vLLM with prefix caching and streaming enabled:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --port 8000
```

The key flag is `--enable-prefix-caching`, which activates the radix-tree-based APC. Without it, every request does a full prefill regardless of shared prefixes.

### Warming the Prefix Cache

To populate the cache, send a request containing the full [A, B] context with minimal generation. This forces vLLM to compute and store the KV blocks for the entire prefix:

```python
warmup_messages = [
    {"role": "system", "content": "You are a helpful research assistant."},
    {"role": "user", "content": f"Here is the first document:\n\n{context_a}"},
    {"role": "assistant", "content": "I've read the first document. Please continue."},
    {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
]

# max_tokens=1 to minimize generation cost — we only care about caching the prefix
response = await client.post("/v1/chat/completions", json={
    "model": model,
    "messages": warmup_messages,
    "max_tokens": 1,
    "temperature": 0.0,
})
```

After this request completes, vLLM's radix tree holds KV blocks for the tokenized form of the full [system, A, assistant, B] sequence.

### Measuring TTFT with Streaming

TTFT is measured by enabling streaming and timing the arrival of the first content token. This isolates the prefill phase (where cache hits matter) from the decode phase:

```python
payload = {
    "model": model,
    "messages": messages,
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": True,
    "stream_options": {"include_usage": True},
}

start = time.perf_counter()
first_token_time = None

async with session.post(url, json=payload) as resp:
    async for line in resp.content:
        chunk = parse_sse(line)
        if chunk and chunk_has_content(chunk) and first_token_time is None:
            first_token_time = time.perf_counter()

ttft_ms = (first_token_time - start) * 1000
```

`stream: True` is essential — without it, the API returns only after full generation completes, making it impossible to separate prefill latency from decode latency.

`stream_options: {"include_usage": True}` gives us accurate `prompt_tokens` and `completion_tokens` counts in the final SSE chunk.

### Query Messages: Cache Hit (T2)

The query appends Q to the same [A, B] prefix that was warmed. vLLM walks its radix tree from root, matches all cached blocks, and only needs to prefill the new tokens (the query itself):

```python
messages_cache_hit = [
    {"role": "system", "content": "You are a helpful research assistant."},
    {"role": "user", "content": f"Here is the first document:\n\n{context_a}"},
    {"role": "assistant", "content": "I've read the first document. Please continue."},
    {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
    {"role": "assistant", "content": "I've read both documents. What would you like to know?"},
    {"role": "user", "content": query},
]
```

### Query Messages: Cache Miss (T3)

The query sends [B, Q] without A. The prefix has changed from position 0, so no cached blocks match:

```python
messages_cache_miss = [
    {"role": "system", "content": "You are a helpful research assistant."},
    {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
    {"role": "assistant", "content": "I've read the document. What would you like to know?"},
    {"role": "user", "content": query},
]
```

### Trial Structure

Each condition (T2 and T3) is measured across multiple trials with warmup:

```
For each trial:
  1. Warm the prefix cache with [A, B]       ← ensures cache is populated
  2. Wait 500ms                               ← let cache settle
  3. Send the query (either [A,B,Q] or [B,Q]) ← measure TTFT and throughput
```

Warmup trials are discarded. We report mean, median, and standard deviation of TTFT across measured trials.

## Metrics Collected

| Metric | Description | Why It Matters |
|--------|-------------|---------------|
| TTFT (ms) | Time from request sent to first token received | Directly reflects prefill cost; cache hits reduce this |
| Total time (ms) | Time from request sent to stream completion | End-to-end latency including decode |
| Throughput (tok/s) | Output tokens / decode time | Shows if cache miss also impacts decode scheduling |
| Prompt tokens | Token count reported by vLLM | Verifies context sizes are as expected |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_size_a` | 4096 tokens | Size of the removable context block A |
| `context_size_b` | 2048 tokens | Size of the retained context block B |
| `max_output_tokens` | 128 | Generation length for query Q |
| `num_trials` | 5 | Measured trials per condition |
| `warmup_trials` | 1 | Discarded warmup trials per condition |