"""
Phase 2: Prefix Cache Invalidation Cost — Direct vLLM Python API

Measures TTFT via single-token generation using the vLLM LLM class directly
(no HTTP server). Automatic prefix caching is enabled so the same radix-tree
APC logic applies.

Compares:
  T2 (cache hit)  — prompt with full [A, B, Q] prefix (matches warmed cache)
  T3 (cache miss)  — prompt with [B, Q] only (A removed, prefix changed)

Outputs results to results_phase2_a{A}_b{B}.json.
"""

import argparse
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict

from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Configuration (defaults, overridable via CLI)
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/opt-125m"

CONTEXT_SIZE_A = 512
CONTEXT_SIZE_B = 256
MAX_OUTPUT_TOKENS = 50
NUM_TRIALS = 10
WARMUP_TOKENS = 6


# ---------------------------------------------------------------------------
# Helpers
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


def build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """Concatenate message contents — mirrors the server chat template."""
    return "".join(msg["content"] for msg in messages)


def compute_stats(trials: List[Dict]) -> Dict:
    """Compute summary statistics from a list of trial result dicts."""
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
# Message builders
# ---------------------------------------------------------------------------

def build_warmup_messages(context_a: str, context_b: str) -> List[Dict[str, str]]:
    """Build the [A, B] warmup prompt messages."""
    return [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"Here is the first document:\n\n{context_a}"},
        {"role": "assistant", "content": "I've read the first document. Please continue."},
        {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
    ]


def build_hit_messages(context_a: str, context_b: str, query: str) -> List[Dict[str, str]]:
    """Build the cache-hit prompt: [A, B, Q] (extends warmed prefix)."""
    return [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"Here is the first document:\n\n{context_a}"},
        {"role": "assistant", "content": "I've read the first document. Please continue."},
        {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
        {"role": "assistant", "content": "I've read both documents. What would you like to know?"},
        {"role": "user", "content": query},
    ]


def build_miss_messages(context_b: str, query: str) -> List[Dict[str, str]]:
    """Build the cache-miss prompt: [B, Q] only (A removed)."""
    return [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
        {"role": "assistant", "content": "I've read the document. What would you like to know?"},
        {"role": "user", "content": query},
    ]


# ---------------------------------------------------------------------------
# Trial runner (pure measurement — no warmup)
# ---------------------------------------------------------------------------

def run_trial(
    llm: LLM,
    query_prompt: str,
) -> Dict[str, float]:
    """
    Single trial — pure measurement, no warmup or cache manipulation.

    Steps:
      1. Measure TTFT (1-token generation)
      2. Sleep 500ms
      3. Measure throughput (50-token generation)
    """
    single_token = SamplingParams(max_tokens=1, temperature=0.0)
    full_gen = SamplingParams(max_tokens=MAX_OUTPUT_TOKENS, temperature=0.0)

    # Step 1: TTFT measurement (1-token generation)
    start = time.perf_counter()
    ttft_output = llm.generate([query_prompt], single_token, use_tqdm=False)
    ttft_ms = (time.perf_counter() - start) * 1000
    prompt_tokens = len(ttft_output[0].prompt_token_ids)

    # Step 2: settle
    time.sleep(0.5)

    # Step 3: throughput measurement (full generation)
    start = time.perf_counter()
    full_output = llm.generate([query_prompt], full_gen, use_tqdm=False)
    total_time_ms = (time.perf_counter() - start) * 1000
    completion_tokens = len(full_output[0].outputs[0].token_ids)
    throughput = completion_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0

    return {
        "ttft_ms": ttft_ms,
        "total_time_ms": total_time_ms,
        "throughput": throughput,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 80)
    print("Experiment 1 — Phase 2: Direct vLLM Python API")
    print("=" * 80)
    print()

    # ---- Load model ----
    print("Initializing model...")
    llm = LLM(
        model=MODEL_NAME,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.7,
        max_model_len=1024,
        enforce_eager=True,
        swap_space=0,
    )
    print("Model loaded.\n")

    # ---- Build contexts and prompts ----
    context_a = generate_context(CONTEXT_SIZE_A, "A")
    context_b = generate_context(CONTEXT_SIZE_B, "B")
    query = "Summarize the key points from the documents."

    warmup_prompt = build_prompt_from_messages(build_warmup_messages(context_a, context_b))
    hit_prompt = build_prompt_from_messages(build_hit_messages(context_a, context_b, query))
    miss_prompt = build_prompt_from_messages(build_miss_messages(context_b, query))

    print(f"Context A: ~{CONTEXT_SIZE_A} tokens")
    print(f"Context B: ~{CONTEXT_SIZE_B} tokens")
    print(f"Max output tokens: {MAX_OUTPUT_TOKENS}")
    print(f"Trials: {NUM_TRIALS} measured")
    print()

    # ---- ONE-TIME warmup: send [A,B] with 6 tokens ----
    warmup_params = SamplingParams(max_tokens=WARMUP_TOKENS, temperature=0.0)
    print(f"Warming cache: generating {WARMUP_TOKENS} tokens with [A,B] prompt...")
    warmup_start = time.perf_counter()
    warmup_output = llm.generate([warmup_prompt], warmup_params, use_tqdm=False)
    warmup_time_ms = (time.perf_counter() - warmup_start) * 1000
    warmup_gen_tokens = len(warmup_output[0].outputs[0].token_ids)
    print(f"Warmup done: {warmup_time_ms:.2f}ms, generated {warmup_gen_tokens} tokens\n")

    # Settle after warmup
    time.sleep(0.5)

    # ---- Cache hit trials (T2) ----
    print("Running CACHE HIT trials...")
    cache_hit_results: List[Dict] = []
    for i in range(NUM_TRIALS):
        result = run_trial(llm, hit_prompt)
        cache_hit_results.append(result)
        print(
            f"  Trial {i + 1}: TTFT={result['ttft_ms']:.2f}ms  "
            f"Total={result['total_time_ms']:.2f}ms  "
            f"Throughput={result['throughput']:.2f} tok/s"
        )
    print()

    # ---- Cache miss trials (T3) ----
    print("Running CACHE MISS trials...")
    cache_miss_results: List[Dict] = []
    for i in range(NUM_TRIALS):
        result = run_trial(llm, miss_prompt)
        cache_miss_results.append(result)
        print(
            f"  Trial {i + 1}: TTFT={result['ttft_ms']:.2f}ms  "
            f"Total={result['total_time_ms']:.2f}ms  "
            f"Throughput={result['throughput']:.2f} tok/s"
        )
    print()

    # ---- Statistics ----
    hit_stats = compute_stats(cache_hit_results)
    miss_stats = compute_stats(cache_miss_results)
    # Inject warmup into both conditions so plot_results.py shows the warmup panel
    for stats in (hit_stats, miss_stats):
        stats["warmup_time_mean"] = warmup_time_ms
        stats["warmup_time_stdev"] = 0.0
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
        "phase": "phase2_python_api",
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "context_size_a": CONTEXT_SIZE_A,
            "context_size_b": CONTEXT_SIZE_B,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "num_trials": NUM_TRIALS,
        },
        "warmup_time_ms": warmup_time_ms,
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

    outfile = f"results_phase2_a{CONTEXT_SIZE_A}_b{CONTEXT_SIZE_B}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Direct vLLM API experiment")
    parser.add_argument("--size-a", type=int, default=CONTEXT_SIZE_A, help="Context A token count")
    parser.add_argument("--size-b", type=int, default=CONTEXT_SIZE_B, help="Context B token count")
    args = parser.parse_args()
    CONTEXT_SIZE_A = args.size_a
    CONTEXT_SIZE_B = args.size_b
    run_experiment()
