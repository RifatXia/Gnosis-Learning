"""
Phase 1: Prefix Cache Invalidation Cost — HTTP Streaming API

Measures TTFT (time to first token) via SSE streaming against a vLLM server
with automatic prefix caching enabled.

Compares:
  T2 (cache hit)  — query with full [A, B, Q] prefix (matches warmed cache)
  T3 (cache miss)  — query with [B, Q] only (A removed, prefix changed)

Outputs results to results_phase1.json.
"""

import argparse
import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Configuration (defaults, overridable via CLI)
# ---------------------------------------------------------------------------
VLLM_BASE_URL = "http://localhost:8000"
MODEL_NAME = "facebook/opt-125m"

CONTEXT_SIZE_A = 512   # tokens for removable context block A
CONTEXT_SIZE_B = 256   # tokens for retained context block B
MAX_OUTPUT_TOKENS = 50
NUM_TRIALS = 10
WARMUP_TRIALS = 1


# ---------------------------------------------------------------------------
# Helpers (ported from exp_1.py)
# ---------------------------------------------------------------------------

def generate_context(target_tokens: int, label: str) -> str:
    """Generate synthetic context that approximates *target_tokens* tokens."""
    chars_per_token = 4
    target_chars = target_tokens * chars_per_token
    base_text = (
        f"This is document {label}. "
        f"It contains important information about topic {label}. "
    )
    repetitions = target_chars // len(base_text) + 1
    return (base_text * repetitions)[:target_chars]


def parse_sse_line(line: bytes) -> Optional[Dict]:
    """Parse a single server-sent event line."""
    line_str = line.decode("utf-8").strip()
    if not line_str or line_str.startswith(":"):
        return None
    if line_str.startswith("data: "):
        data_str = line_str[6:]
        if data_str == "[DONE]":
            return {"done": True}
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            return None
    return None


def chunk_has_content(chunk: Dict) -> bool:
    """Return True when a streaming chunk carries an actual content token."""
    if "choices" not in chunk:
        return False
    for choice in chunk["choices"]:
        delta = choice.get("delta", {})
        if "content" in delta and delta["content"]:
            return True
    return False


# ---------------------------------------------------------------------------
# Streaming measurement
# ---------------------------------------------------------------------------

async def measure_streaming_request(
    session: aiohttp.ClientSession,
    messages: List[Dict[str, str]],
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> Dict[str, float]:
    """Send a streaming chat-completion request and return timing metrics."""
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start_time = time.perf_counter()
    first_token_time = None
    prompt_tokens = 0
    completion_tokens = 0

    async with session.post(url, json=payload) as resp:
        async for line in resp.content:
            chunk = parse_sse_line(line)
            if chunk is None:
                continue
            if chunk.get("done"):
                break
            if chunk_has_content(chunk) and first_token_time is None:
                first_token_time = time.perf_counter()
            if "usage" in chunk:
                prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                completion_tokens = chunk["usage"].get("completion_tokens", 0)

    end_time = time.perf_counter()

    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    total_time_ms = (end_time - start_time) * 1000
    decode_time_s = (end_time - first_token_time) if first_token_time else 0.001
    throughput = completion_tokens / decode_time_s if decode_time_s > 0 else 0

    return {
        "ttft_ms": ttft_ms,
        "total_time_ms": total_time_ms,
        "throughput": throughput,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ---------------------------------------------------------------------------
# Cache warming & trial execution
# ---------------------------------------------------------------------------

async def warm_prefix_cache(
    session: aiohttp.ClientSession,
    context_a: str,
    context_b: str,
) -> None:
    """Populate the prefix cache by sending a request with the full [A, B] context."""
    warmup_messages = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"Here is the first document:\n\n{context_a}"},
        {"role": "assistant", "content": "I've read the first document. Please continue."},
        {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
    ]
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": warmup_messages,
        "max_tokens": 1,
        "temperature": 0.0,
    }
    async with session.post(url, json=payload) as resp:
        await resp.json()


async def run_trial(
    session: aiohttp.ClientSession,
    context_a: str,
    context_b: str,
    query: str,
    use_cache_hit: bool,
) -> Dict[str, float]:
    """Run one trial: warm cache → wait → measure query."""
    await warm_prefix_cache(session, context_a, context_b)
    await asyncio.sleep(0.5)

    if use_cache_hit:
        messages = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": f"Here is the first document:\n\n{context_a}"},
            {"role": "assistant", "content": "I've read the first document. Please continue."},
            {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
            {"role": "assistant", "content": "I've read both documents. What would you like to know?"},
            {"role": "user", "content": query},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
            {"role": "assistant", "content": "I've read the document. What would you like to know?"},
            {"role": "user", "content": query},
        ]

    return await measure_streaming_request(session, messages, MAX_OUTPUT_TOKENS)


# ---------------------------------------------------------------------------
# Server readiness check
# ---------------------------------------------------------------------------

async def wait_for_server(max_retries: int = 30, delay: int = 2) -> bool:
    """Block until the vLLM health endpoint responds 200."""
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=5)
                async with session.get(f"{VLLM_BASE_URL}/health", timeout=timeout) as resp:
                    if resp.status == 200:
                        print("Server is ready")
                        return True
        except Exception:
            pass
        if attempt < max_retries - 1:
            print(f"  waiting for vLLM server... ({attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)

    print("vLLM server did not start in time")
    return False


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def compute_stats(trials: List[Dict]) -> Dict:
    ttft = [t["ttft_ms"] for t in trials]
    total = [t["total_time_ms"] for t in trials]
    tput = [t["throughput"] for t in trials]
    return {
        "ttft_mean": statistics.mean(ttft),
        "ttft_median": statistics.median(ttft),
        "ttft_stdev": statistics.stdev(ttft),
        "total_time_mean": statistics.mean(total),
        "total_time_stdev": statistics.stdev(total),
        "throughput_mean": statistics.mean(tput),
        "throughput_stdev": statistics.stdev(tput),
        "prompt_tokens": trials[0]["prompt_tokens"],
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def run_experiment():
    print("=" * 80)
    print("Experiment 1 — Phase 1: HTTP Streaming API")
    print("=" * 80)
    print()

    print("Checking vLLM server...")
    if not await wait_for_server():
        print("Error: vLLM server is not running on port 8000")
        return
    print()

    context_a = generate_context(CONTEXT_SIZE_A, "A")
    context_b = generate_context(CONTEXT_SIZE_B, "B")
    query = "Summarize the key points from the documents."

    print(f"Context A: ~{CONTEXT_SIZE_A} tokens")
    print(f"Context B: ~{CONTEXT_SIZE_B} tokens")
    print(f"Max output tokens: {MAX_OUTPUT_TOKENS}")
    print(f"Trials: {WARMUP_TRIALS} warmup + {NUM_TRIALS} measured")
    print()

    async with aiohttp.ClientSession() as session:
        # ---- Cache hit trials (T2) ----
        print("Running CACHE HIT trials...")
        cache_hit_results: List[Dict] = []
        for i in range(WARMUP_TRIALS + NUM_TRIALS):
            result = await run_trial(session, context_a, context_b, query, use_cache_hit=True)
            if i >= WARMUP_TRIALS:
                cache_hit_results.append(result)
                idx = i - WARMUP_TRIALS + 1
                print(
                    f"  Trial {idx}: TTFT={result['ttft_ms']:.2f}ms  "
                    f"Total={result['total_time_ms']:.2f}ms  "
                    f"Throughput={result['throughput']:.2f} tok/s"
                )
        print()

        # ---- Cache miss trials (T3) ----
        print("Running CACHE MISS trials...")
        cache_miss_results: List[Dict] = []
        for i in range(WARMUP_TRIALS + NUM_TRIALS):
            result = await run_trial(session, context_a, context_b, query, use_cache_hit=False)
            if i >= WARMUP_TRIALS:
                cache_miss_results.append(result)
                idx = i - WARMUP_TRIALS + 1
                print(
                    f"  Trial {idx}: TTFT={result['ttft_ms']:.2f}ms  "
                    f"Total={result['total_time_ms']:.2f}ms  "
                    f"Throughput={result['throughput']:.2f} tok/s"
                )
        print()

    # ---- Statistics ----
    hit_stats = compute_stats(cache_hit_results)
    miss_stats = compute_stats(cache_miss_results)
    ttft_degradation = (miss_stats["ttft_mean"] / hit_stats["ttft_mean"] - 1) * 100

    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    print("CACHE HIT (T2) — Full prefix [A, B] cached:")
    print(f"  TTFT:       mean={hit_stats['ttft_mean']:.2f}ms  "
          f"median={hit_stats['ttft_median']:.2f}ms  "
          f"stdev={hit_stats['ttft_stdev']:.2f}ms")
    print(f"  Total Time: mean={hit_stats['total_time_mean']:.2f}ms  "
          f"stdev={hit_stats['total_time_stdev']:.2f}ms")
    print(f"  Throughput: mean={hit_stats['throughput_mean']:.2f} tok/s  "
          f"stdev={hit_stats['throughput_stdev']:.2f}")
    print(f"  Prompt tokens: {hit_stats['prompt_tokens']}")
    print()
    print("CACHE MISS (T3) — Prefix [B] only, A removed:")
    print(f"  TTFT:       mean={miss_stats['ttft_mean']:.2f}ms  "
          f"median={miss_stats['ttft_median']:.2f}ms  "
          f"stdev={miss_stats['ttft_stdev']:.2f}ms")
    print(f"  Total Time: mean={miss_stats['total_time_mean']:.2f}ms  "
          f"stdev={miss_stats['total_time_stdev']:.2f}ms")
    print(f"  Throughput: mean={miss_stats['throughput_mean']:.2f} tok/s  "
          f"stdev={miss_stats['throughput_stdev']:.2f}")
    print(f"  Prompt tokens: {miss_stats['prompt_tokens']}")
    print()
    print(f"TTFT Degradation: {ttft_degradation:.1f}% slower on cache miss")
    print("=" * 80)

    # ---- Save JSON ----
    output = {
        "phase": "phase1_http_api",
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "context_size_a": CONTEXT_SIZE_A,
            "context_size_b": CONTEXT_SIZE_B,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "num_trials": NUM_TRIALS,
            "warmup_trials": WARMUP_TRIALS,
        },
        "cache_hit": {
            "trials": cache_hit_results,
            "stats": hit_stats,
        },
        "cache_miss": {
            "trials": cache_miss_results,
            "stats": miss_stats,
        },
        "ttft_degradation_pct": ttft_degradation,
    }

    outfile = f"results_phase1_a{CONTEXT_SIZE_A}_b{CONTEXT_SIZE_B}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: HTTP streaming experiment")
    parser.add_argument("--size-a", type=int, default=CONTEXT_SIZE_A, help="Context A token count")
    parser.add_argument("--size-b", type=int, default=CONTEXT_SIZE_B, help="Context B token count")
    args = parser.parse_args()
    CONTEXT_SIZE_A = args.size_a
    CONTEXT_SIZE_B = args.size_b
    asyncio.run(run_experiment())
