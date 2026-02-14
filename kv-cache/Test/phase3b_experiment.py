"""
Phase 3b: Prefix Cache Invalidation Cost — llama-cpp-python Direct API

Measures TTFT via single-token generation using the llama-cpp-python Llama class
directly (no HTTP server). KV cache is explicitly cleared between trials to
prevent contamination.

Flow (repeated NUM_TRIALS times):
  1. Clear cache -> Warmup [A,B] -> Ask [A,B,Q] (cache hit)
  2. Clear cache -> Warmup [A,B] -> Ask [B,Q] (cache miss)

This gives 20 warmups total (10 for hit, 10 for miss) with std deviation.

Compares:
  T2 (cache hit)  — prompt with full [A, B, Q] prefix (matches warmed cache)
  T3 (cache miss)  — prompt with [B, Q] only (A removed, prefix changed)

Outputs results to results_phase3b_a{A}_b{B}.json.
"""

import argparse
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict

from llama_cpp import Llama


# ---------------------------------------------------------------------------
# Configuration (defaults, overridable via CLI)
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/rifatxia/Desktop/Gnosis/Gnosis-Learning/nsight-systems/llama.cpp/models/gpt2.gguf"
MODEL_NAME = "gpt2"

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


def clear_kv_cache(llm: Llama):
    """Clear the KV cache to ensure no contamination between trials."""
    llm.reset()


def compute_stats(trials: List[Dict]) -> Dict:
    """Compute summary statistics from a list of trial result dicts."""
    ttft = [t["ttft_ms"] for t in trials]
    total = [t["total_time_ms"] for t in trials]
    tput = [t["throughput"] for t in trials]
    return {
        "ttft_mean": statistics.mean(ttft),
        "ttft_median": statistics.median(ttft),
        "ttft_stdev": statistics.stdev(ttft) if len(ttft) > 1 else 0.0,
        "total_time_mean": statistics.mean(total),
        "total_time_stdev": statistics.stdev(total) if len(total) > 1 else 0.0,
        "throughput_mean": statistics.mean(tput),
        "throughput_stdev": statistics.stdev(tput) if len(tput) > 1 else 0.0,
        "prompt_tokens": trials[0]["prompt_tokens"],
    }


def compute_warmup_stats(warmup_times: List[float]) -> Dict:
    """Compute warmup statistics from all warmup times (20 total)."""
    return {
        "warmup_time_mean": statistics.mean(warmup_times),
        "warmup_time_median": statistics.median(warmup_times),
        "warmup_time_stdev": statistics.stdev(warmup_times) if len(warmup_times) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def build_warmup_messages(context_a: str, context_b: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"Here is the first document:\n\n{context_a}"},
        {"role": "assistant", "content": "I've read the first document. Please continue."},
        {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
    ]


def build_hit_messages(context_a: str, context_b: str, query: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": f"Here is the first document:\n\n{context_a}"},
        {"role": "assistant", "content": "I've read the first document. Please continue."},
        {"role": "user", "content": f"Here is the second document:\n\n{context_b}"},
        {"role": "assistant", "content": "I've read both documents. What would you like to know?"},
        {"role": "user", "content": query},
    ]


def build_miss_messages(context_b: str, query: str) -> List[Dict[str, str]]:
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
    llm: Llama,
    query_prompt: str,
) -> Dict[str, float]:
    """
    Single trial — pure measurement, no warmup or cache manipulation.

    Steps:
      1. Measure TTFT (1-token generation)
      2. Sleep 500ms
      3. Measure throughput (50-token generation)
    """
    # Step 1: TTFT measurement (1-token generation)
    start = time.perf_counter()
    ttft_output = llm(query_prompt, max_tokens=1, temperature=0.0)
    ttft_ms = (time.perf_counter() - start) * 1000
    prompt_tokens = ttft_output["usage"]["prompt_tokens"]

    # Step 2: settle
    time.sleep(0.5)

    # Step 3: throughput measurement (full generation)
    start = time.perf_counter()
    full_output = llm(query_prompt, max_tokens=MAX_OUTPUT_TOKENS, temperature=0.0)
    total_time_ms = (time.perf_counter() - start) * 1000
    completion_tokens = full_output["usage"]["completion_tokens"]
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
    print("Experiment 1 — Phase 3b: llama-cpp-python Direct API")
    print("=" * 80)
    print()

    print("Initializing model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_batch=2048,
        n_ubatch=512,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("Model loaded.\n")

    context_a = generate_context(CONTEXT_SIZE_A, "A")
    context_b = generate_context(CONTEXT_SIZE_B, "B")
    query = "Summarize the key points from the documents."

    warmup_prompt = build_prompt_from_messages(build_warmup_messages(context_a, context_b))
    hit_prompt = build_prompt_from_messages(build_hit_messages(context_a, context_b, query))
    miss_prompt = build_prompt_from_messages(build_miss_messages(context_b, query))

    print(f"Context A: ~{CONTEXT_SIZE_A} tokens")
    print(f"Context B: ~{CONTEXT_SIZE_B} tokens")
    print(f"Max output tokens: {MAX_OUTPUT_TOKENS}")
    print(f"Trials: {NUM_TRIALS} per condition (separate loops)")
    print(f"Total warmups: {NUM_TRIALS * 2} (20 warmups for std deviation)")
    print()

    # Storage for results
    cache_hit_results: List[Dict] = []
    cache_miss_results: List[Dict] = []
    all_warmup_times: List[float] = []

    # ---- CACHE HIT trials (separate loop) ----
    print("Running CACHE HIT trials...")
    print("-" * 80)
    for trial_idx in range(NUM_TRIALS):
        # Clear -> Warmup -> Ask [A,B,Q]
        clear_kv_cache(llm)

        # Warmup with [A,B]
        warmup_start = time.perf_counter()
        llm(warmup_prompt, max_tokens=WARMUP_TOKENS, temperature=0.0)
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        all_warmup_times.append(warmup_time)

        # Settle
        time.sleep(0.5)

        # Measure cache hit
        hit_result = run_trial(llm, hit_prompt)
        cache_hit_results.append(hit_result)
        print(
            f"  Trial {trial_idx + 1}: Warmup={warmup_time:.2f}ms  "
            f"TTFT={hit_result['ttft_ms']:.2f}ms  "
            f"Total={hit_result['total_time_ms']:.2f}ms  "
            f"Throughput={hit_result['throughput']:.2f} tok/s"
        )

    print()

    # ---- CACHE MISS trials (separate loop) ----
    print("Running CACHE MISS trials...")
    print("-" * 80)
    for trial_idx in range(NUM_TRIALS):
        # Clear -> Warmup -> Ask [B,Q]
        clear_kv_cache(llm)

        # Warmup with [A,B]
        warmup_start = time.perf_counter()
        llm(warmup_prompt, max_tokens=WARMUP_TOKENS, temperature=0.0)
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        all_warmup_times.append(warmup_time)

        # Settle
        time.sleep(0.5)

        # Measure cache miss
        miss_result = run_trial(llm, miss_prompt)
        cache_miss_results.append(miss_result)
        print(
            f"  Trial {trial_idx + 1}: Warmup={warmup_time:.2f}ms  "
            f"TTFT={miss_result['ttft_ms']:.2f}ms  "
            f"Total={miss_result['total_time_ms']:.2f}ms  "
            f"Throughput={miss_result['throughput']:.2f} tok/s"
        )


    print()

    # ---- Statistics ----
    hit_stats = compute_stats(cache_hit_results)
    miss_stats = compute_stats(cache_miss_results)
    warmup_stats = compute_warmup_stats(all_warmup_times)

    # Inject warmup stats into both hit and miss stats for plot_results.py compatibility
    for stats in (hit_stats, miss_stats):
        stats["warmup_time_mean"] = warmup_stats["warmup_time_mean"]
        stats["warmup_time_median"] = warmup_stats["warmup_time_median"]
        stats["warmup_time_stdev"] = warmup_stats["warmup_time_stdev"]

    ttft_degradation = (miss_stats["ttft_mean"] / hit_stats["ttft_mean"] - 1) * 100

    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    print(f"WARMUP (all {len(all_warmup_times)} warmups combined):")
    print(f"  mean={warmup_stats['warmup_time_mean']:.2f}ms  "
          f"median={warmup_stats['warmup_time_median']:.2f}ms  "
          f"stdev={warmup_stats['warmup_time_stdev']:.2f}ms")
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
        "phase": "phase3b_python_api",
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "context_size_a": CONTEXT_SIZE_A,
            "context_size_b": CONTEXT_SIZE_B,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "num_trials": NUM_TRIALS,
        },
        "warmup": {
            "times_ms": all_warmup_times,
            "stats": warmup_stats,
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

    outfile = f"results_phase3b_a{CONTEXT_SIZE_A}_b{CONTEXT_SIZE_B}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3b: llama-cpp-python direct API experiment")
    parser.add_argument("--size-a", type=int, default=CONTEXT_SIZE_A, help="Context A token count")
    parser.add_argument("--size-b", type=int, default=CONTEXT_SIZE_B, help="Context B token count")
    args = parser.parse_args()
    CONTEXT_SIZE_A = args.size_a
    CONTEXT_SIZE_B = args.size_b
    run_experiment()
