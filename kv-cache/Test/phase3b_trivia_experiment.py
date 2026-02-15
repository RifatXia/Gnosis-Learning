"""
Phase 3b Trivia: Prefix Cache Invalidation Cost — TriviaQA Real Text

Identical to phase3b_experiment.py except context A and B come from real
TriviaQA validation-set search passages instead of synthetic filler text.
This lets us verify that KV cache hit/miss behaviour is independent of
prompt content (it should be — caching is purely prefix-token matching).

Flow (repeated NUM_TRIALS times):
  1. Clear cache -> Warmup [A,B] -> Ask [A,B,Q] (cache hit)
  2. Clear cache -> Warmup [A,B] -> Ask [B,Q] (cache miss)

Outputs:
  results/results_phase3b_trivia_a{A}_b{B}.csv  — per-trial metrics (20 rows)
  prompts/prompts_phase3b_trivia_a{A}_b{B}.csv  — prompt text, token counts, sample outputs
"""

import argparse
import csv
import os
import time
import statistics
from typing import List, Dict, Tuple

from datasets import load_dataset
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
# TriviaQA context loader
# ---------------------------------------------------------------------------

def load_trivia_contexts(
    llm: Llama, target_a_tokens: int, target_b_tokens: int
) -> Tuple[str, str, str, str]:
    """Load a TriviaQA validation entry with 2 search passages.

    Iterates until finding an entry where both search_context[0] and [1]
    have enough tokens. Truncates to exact target counts via tokenizer.
    Returns (context_a, context_b, query, question_id).
    """
    ds = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
    for entry in ds:
        contexts = entry["search_results"]["search_context"]
        if len(contexts) < 2:
            continue
        tokens_a = llm.tokenize(contexts[0].encode())
        tokens_b = llm.tokenize(contexts[1].encode())
        if len(tokens_a) >= target_a_tokens and len(tokens_b) >= target_b_tokens:
            trunc_a = llm.detokenize(tokens_a[:target_a_tokens]).decode(errors="replace")
            trunc_b = llm.detokenize(tokens_b[:target_b_tokens]).decode(errors="replace")
            return trunc_a, trunc_b, entry["question"], entry["question_id"]
    raise RuntimeError("No suitable TriviaQA entry found")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """Concatenate message contents — mirrors the server chat template."""
    return "".join(msg["content"] for msg in messages)


def clear_kv_cache(llm: Llama):
    """Fully clear the KV cache — reset state AND clear cache memory."""
    llm.reset()                  # reset n_tokens counter to 0
    llm._ctx.kv_cache_clear()    # actually clear KV cache data from GPU


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
) -> Dict:
    """
    Single trial — pure measurement, no warmup or cache manipulation.

    Steps:
      1. Measure TTFT (1-token generation)
      2. Sleep 500ms
      3. Measure throughput (50-token generation)

    Returns dict with ttft_ms, total_time_ms, throughput, prompt_tokens,
    completion_tokens, finish_reason.
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

    # Determine finish_reason from the output
    finish_reason = "length"
    if "choices" in full_output and full_output["choices"]:
        finish_reason = full_output["choices"][0].get("finish_reason", "length")

    # Extract generated text for sample logging
    generated_text = ""
    if "choices" in full_output and full_output["choices"]:
        generated_text = full_output["choices"][0].get("text", "")

    return {
        "ttft_ms": ttft_ms,
        "total_time_ms": total_time_ms,
        "throughput": throughput,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "generated_text": generated_text,
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

RESULTS_COLUMNS = [
    "trial", "condition", "warmup_time_ms", "ttft_ms", "total_time_ms",
    "throughput_tok_s", "prompt_tokens", "completion_tokens", "finish_reason",
]

PROMPTS_COLUMNS = [
    "prompt_name", "token_count", "prompt_text", "sample_output",
]


def write_results_csv(path: str, rows: List[Dict]):
    """Write per-trial results CSV with flush+fsync."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def write_prompts_csv(path: str, rows: List[Dict]):
    """Write prompts CSV with flush+fsync."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROMPTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 80)
    print("Experiment 1 — Phase 3b Trivia: TriviaQA Real Text")
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

    print("Loading TriviaQA validation passages...")
    context_a, context_b, query, question_id = load_trivia_contexts(
        llm, CONTEXT_SIZE_A, CONTEXT_SIZE_B
    )
    print(f"  Question ID: {question_id}")
    print(f"  Question:    {query}")
    print(f"  Passage A:   {context_a[:100]}...")
    print(f"  Passage B:   {context_b[:100]}...")
    print()

    warmup_prompt = build_prompt_from_messages(build_warmup_messages(context_a, context_b))
    hit_prompt = build_prompt_from_messages(build_hit_messages(context_a, context_b, query))
    miss_prompt = build_prompt_from_messages(build_miss_messages(context_b, query))

    # Tokenize prompts to get exact token counts
    warmup_token_count = len(llm.tokenize(warmup_prompt.encode()))
    hit_token_count = len(llm.tokenize(hit_prompt.encode()))
    miss_token_count = len(llm.tokenize(miss_prompt.encode()))

    print(f"Context A: ~{CONTEXT_SIZE_A} tokens")
    print(f"Context B: ~{CONTEXT_SIZE_B} tokens")
    print(f"Max output tokens: {MAX_OUTPUT_TOKENS}")
    print(f"Trials: {NUM_TRIALS} per condition")
    print(f"Total warmups: {NUM_TRIALS * 2} (for std deviation)")
    print()
    print("Prompt token counts:")
    print(f"  warmup: {warmup_token_count} tokens")
    print(f"  hit:    {hit_token_count} tokens")
    print(f"  miss:   {miss_token_count} tokens")
    print()

    # Storage for results (all in memory — no I/O during timing)
    cache_hit_results: List[Dict] = []
    cache_miss_results: List[Dict] = []
    hit_warmup_times: List[float] = []
    miss_warmup_times: List[float] = []
    csv_rows: List[Dict] = []

    # Sample outputs for prompts CSV (captured from first trial)
    sample_warmup_output = ""
    sample_hit_output = ""
    sample_miss_output = ""

    eos_warnings: List[str] = []

    # ---- CACHE HIT trials ----
    print("Running CACHE HIT trials...")
    print("-" * 80)
    for trial_idx in range(NUM_TRIALS):
        # Clear -> Warmup -> Ask [A,B,Q]
        clear_kv_cache(llm)

        # Warmup with [A,B]
        warmup_start = time.perf_counter()
        warmup_output = llm(warmup_prompt, max_tokens=WARMUP_TOKENS, temperature=0.0)
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        hit_warmup_times.append(warmup_time)

        # Capture sample warmup output from first trial
        if trial_idx == 0 and "choices" in warmup_output and warmup_output["choices"]:
            sample_warmup_output = warmup_output["choices"][0].get("text", "")

        # Settle
        time.sleep(0.5)

        # Measure cache hit
        hit_result = run_trial(llm, hit_prompt)
        cache_hit_results.append(hit_result)

        # Capture sample hit output from first trial
        if trial_idx == 0:
            sample_hit_output = hit_result["generated_text"]

        # Check for early EOS
        if hit_result["completion_tokens"] < MAX_OUTPUT_TOKENS:
            warn = (f"  WARNING: Trial {trial_idx + 1} (hit) — EOS hit early: "
                    f"completion_tokens={hit_result['completion_tokens']} < {MAX_OUTPUT_TOKENS}, "
                    f"finish_reason={hit_result['finish_reason']}")
            eos_warnings.append(warn)
            print(warn)

        # Store CSV row
        csv_rows.append({
            "trial": trial_idx + 1,
            "condition": "hit",
            "warmup_time_ms": f"{warmup_time:.4f}",
            "ttft_ms": f"{hit_result['ttft_ms']:.4f}",
            "total_time_ms": f"{hit_result['total_time_ms']:.4f}",
            "throughput_tok_s": f"{hit_result['throughput']:.4f}",
            "prompt_tokens": hit_result["prompt_tokens"],
            "completion_tokens": hit_result["completion_tokens"],
            "finish_reason": hit_result["finish_reason"],
        })

        print(
            f"  Trial {trial_idx + 1}: Warmup={warmup_time:.2f}ms  "
            f"TTFT={hit_result['ttft_ms']:.2f}ms  "
            f"Total={hit_result['total_time_ms']:.2f}ms  "
            f"Throughput={hit_result['throughput']:.2f} tok/s  "
            f"Tokens={hit_result['completion_tokens']}/{MAX_OUTPUT_TOKENS}  "
            f"Finish={hit_result['finish_reason']}"
        )

    print()

    # ---- CACHE MISS trials ----
    print("Running CACHE MISS trials...")
    print("-" * 80)
    for trial_idx in range(NUM_TRIALS):
        # Clear -> Warmup -> Ask [B,Q]
        clear_kv_cache(llm)

        # Warmup with [A,B]
        warmup_start = time.perf_counter()
        llm(warmup_prompt, max_tokens=WARMUP_TOKENS, temperature=0.0)
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        miss_warmup_times.append(warmup_time)

        # Settle
        time.sleep(0.5)

        # Measure cache miss
        miss_result = run_trial(llm, miss_prompt)
        cache_miss_results.append(miss_result)

        # Capture sample miss output from first trial
        if trial_idx == 0:
            sample_miss_output = miss_result["generated_text"]

        # Check for early EOS
        if miss_result["completion_tokens"] < MAX_OUTPUT_TOKENS:
            warn = (f"  WARNING: Trial {trial_idx + 1} (miss) — EOS hit early: "
                    f"completion_tokens={miss_result['completion_tokens']} < {MAX_OUTPUT_TOKENS}, "
                    f"finish_reason={miss_result['finish_reason']}")
            eos_warnings.append(warn)
            print(warn)

        # Store CSV row
        csv_rows.append({
            "trial": trial_idx + 1,
            "condition": "miss",
            "warmup_time_ms": f"{warmup_time:.4f}",
            "ttft_ms": f"{miss_result['ttft_ms']:.4f}",
            "total_time_ms": f"{miss_result['total_time_ms']:.4f}",
            "throughput_tok_s": f"{miss_result['throughput']:.4f}",
            "prompt_tokens": miss_result["prompt_tokens"],
            "completion_tokens": miss_result["completion_tokens"],
            "finish_reason": miss_result["finish_reason"],
        })

        print(
            f"  Trial {trial_idx + 1}: Warmup={warmup_time:.2f}ms  "
            f"TTFT={miss_result['ttft_ms']:.2f}ms  "
            f"Total={miss_result['total_time_ms']:.2f}ms  "
            f"Throughput={miss_result['throughput']:.2f} tok/s  "
            f"Tokens={miss_result['completion_tokens']}/{MAX_OUTPUT_TOKENS}  "
            f"Finish={miss_result['finish_reason']}"
        )

    print()

    # ---- Statistics ----
    hit_stats = compute_stats(cache_hit_results)
    miss_stats = compute_stats(cache_miss_results)
    all_warmup_times = hit_warmup_times + miss_warmup_times
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
    print(f"TriviaQA Question: {query}")
    print(f"Question ID: {question_id}")
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

    if eos_warnings:
        print()
        print(f"EOS WARNINGS ({len(eos_warnings)} total):")
        for w in eos_warnings:
            print(w)

    print("=" * 80)

    # ---- Write results CSV ----
    results_file = f"results/results_phase3b_trivia_a{CONTEXT_SIZE_A}_b{CONTEXT_SIZE_B}.csv"
    write_results_csv(results_file, csv_rows)
    print(f"\nResults saved to {results_file}")

    # ---- Write prompts CSV ----
    prompts_file = f"prompts/prompts_phase3b_trivia_a{CONTEXT_SIZE_A}_b{CONTEXT_SIZE_B}.csv"
    prompts_rows = [
        {
            "prompt_name": "warmup",
            "token_count": warmup_token_count,
            "prompt_text": warmup_prompt,
            "sample_output": sample_warmup_output,
        },
        {
            "prompt_name": "hit",
            "token_count": hit_token_count,
            "prompt_text": hit_prompt,
            "sample_output": sample_hit_output,
        },
        {
            "prompt_name": "miss",
            "token_count": miss_token_count,
            "prompt_text": miss_prompt,
            "sample_output": sample_miss_output,
        },
    ]
    write_prompts_csv(prompts_file, prompts_rows)
    print(f"Prompts saved to {prompts_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3b Trivia: TriviaQA real-text experiment")
    parser.add_argument("--size-a", type=int, default=CONTEXT_SIZE_A, help="Context A token count")
    parser.add_argument("--size-b", type=int, default=CONTEXT_SIZE_B, help="Context B token count")
    args = parser.parse_args()
    CONTEXT_SIZE_A = args.size_a
    CONTEXT_SIZE_B = args.size_b
    run_experiment()
