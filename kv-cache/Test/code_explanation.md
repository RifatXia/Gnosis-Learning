# Phase 2 Experiment — Code Explanation

Documentation for `phase2_experiment.py` and the plotting pipeline.

---

## Global Variables

| Variable | Purpose |
|----------|---------|
| `MODEL_NAME` | HuggingFace model ID to load (`facebook/opt-125m`) |
| `CONTEXT_SIZE_A` | Target token count for context document A |
| `CONTEXT_SIZE_B` | Target token count for context document B |
| `MAX_OUTPUT_TOKENS` | Max tokens generated in throughput measurement (50) |
| `NUM_TRIALS` | Number of measured trials per condition (10) |
| `WARMUP_TRIALS` | Number of discarded trials before measurement (1) |
| `WARMUP_TOKENS` | Tokens generated during one-time cache warmup (6) |

---

## Functions

### Helpers

| Function | Purpose |
|----------|---------|
| `generate_context(target_tokens, label)` | Creates synthetic text approximating a target token count |
| `build_prompt_from_messages(messages)` | Concatenates message contents into a single prompt string |
| `compute_stats(trials)` | Computes mean/median/stdev for TTFT, total time, throughput |

### Message Builders

| Function | Purpose |
|----------|---------|
| `build_warmup_messages(context_a, context_b)` | Builds [A, B] prompt to populate the prefix cache |
| `build_hit_messages(context_a, context_b, query)` | Builds [A, B, Q] prompt that reuses the cached prefix |
| `build_miss_messages(context_b, query)` | Builds [B, Q] prompt that misses the cached prefix |

### Core

| Function | Purpose |
|----------|---------|
| `run_trial(llm, query_prompt)` | Pure measurement: TTFT (1-token) + throughput (50-token) |
| `run_experiment()` | Orchestrates the full experiment end-to-end |

---

## Entry Point

```
__main__ → parse CLI args (--size-a, --size-b) → run_experiment()
```

CLI overrides `CONTEXT_SIZE_A` and `CONTEXT_SIZE_B` globals before calling `run_experiment()`.

---

## Experiment Workflow

```
run_experiment()
│
├─ 1. Load model (vLLM LLM with prefix caching enabled)
│
├─ 2. Build contexts A, B and all prompt strings
│     ├─ warmup_prompt  → [A, B]
│     ├─ hit_prompt     → [A, B, Q]
│     └─ miss_prompt    → [B, Q]
│
├─ 3. ONE-TIME warmup
│     └─ Generate 6 tokens with [A, B] prompt → populates prefix cache
│     └─ Record warmup_time_ms
│     └─ Sleep 500ms
│
├─ 4. Cache HIT trials (1 discarded + 10 measured)
│     └─ Each trial: TTFT (1-token) → sleep 500ms → throughput (50-token)
│
├─ 5. Cache MISS trials (1 discarded + 10 measured)
│     └─ Each trial: TTFT (1-token) → sleep 500ms → throughput (50-token)
│
├─ 6. Compute stats (mean, median, stdev for each metric)
│
├─ 7. Print summary to console
│
└─ 8. Save JSON → results_phase2_a{A}_b{B}.json
```

### Trial Details (`run_trial`)

Each trial is a **pure measurement** — no warmup or cache manipulation:
1. Generate 1 token with the query prompt → measures TTFT
2. Sleep 500ms (let GPU settle)
3. Generate 50 tokens with the query prompt → measures throughput

---

## JSON Output Schema

```json
{
  "phase": "phase2_python_api",
  "model": "facebook/opt-125m",
  "timestamp": "...",
  "config": {
    "context_size_a": 512,
    "context_size_b": 256,
    "max_output_tokens": 50,
    "num_trials": 10,
    "warmup_trials": 1
  },
  "warmup_time_ms": 123.45,          // top-level, single value
  "cache_hit": {
    "trials": [ ... ],               // per-trial raw measurements
    "stats": { "ttft_mean", "ttft_median", "ttft_stdev",
               "total_time_mean", "total_time_stdev",
               "throughput_mean", "throughput_stdev",
               "prompt_tokens" }
  },
  "cache_miss": { ... },             // same structure as cache_hit
  "ttft_degradation_pct": 15.3
}
```

---

## Plot Generation Pipeline

```
plot_results.py
│
├─ discover_results()
│     └─ Glob for results_phase*_a*_b*.json → group by (A, B) and phase
│
├─ plot_phase(configs, phase_key, phase_label, outfile)
│     ├─ Checks if "warmup_time_mean" exists in per-condition stats
│     │   └─ Phase 2: NO → has_warmup = False → 2-panel plot
│     ├─ Panel 1: TTFT grouped bar chart (hit vs miss × config)
│     └─ Panel 2: Throughput grouped bar chart (hit vs miss × config)
│
└─ Output: experiment1_all_configs_phase2.png
```

Since `warmup_time_ms` is now at the top level (not in per-condition stats), `plot_results.py` produces a 2-panel plot (TTFT + Throughput) with no warmup panel.
