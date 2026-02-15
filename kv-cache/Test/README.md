# Experiment 1: Prefix Cache Invalidation Cost

Measures the performance cost of KV cache invalidation when context is removed from a multi-turn conversation served by vLLM with automatic prefix caching (APC) enabled.

## Hypothesis

When a prefix changes (context A removed), vLLM's radix-tree APC cannot reuse cached KV blocks, forcing full recomputation. This should produce measurably higher TTFT compared to a cache-hit scenario where the full prefix is preserved.

## Setup

| Component | Value |
|-----------|-------|
| Model | `facebook/opt-125m` (~250 MB) |
| GPU | NVIDIA GeForce GTX 1650 (4 GB VRAM) |
| vLLM | 0.6.3.post1 with `--enable-prefix-caching` |
| Context A | ~512 tokens (removable block) |
| Context B | ~256 tokens (retained block) |
| Output | 50 tokens max |
| Trials | 10 measured + 1 warmup |

## Experiment Design

Each trial follows the same pattern:

1. **Warm cache** — send `[A, B]` prompt with `max_tokens=1` to populate APC
2. **Wait 500 ms** — let cache state settle
3. **Measure query** — timed request with the test prompt

Two conditions are compared:

- **T2 (Cache Hit)** — query uses full `[A, B, Q]` prefix, matching the warmed cache
- **T3 (Cache Miss)** — query uses `[B, Q]` only (A removed), invalidating the prefix

### Phase 1: HTTP Streaming API

Uses vLLM's OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming. TTFT is measured as the wall-clock time from request send to first content token received.

### Phase 2: Direct Python API

Uses vLLM's `LLM` class directly (no HTTP server). TTFT is measured as the wall-clock time for a 1-token generation call (`prefill + 1 decode step`). Since the single decode step is constant across conditions, the T2 vs T3 delta isolates the cache effect.

## File Structure

```
KV Cache/Test/
├── pyproject.toml            # uv project config + dependencies
├── phase1_server.sh          # Launches vLLM HTTP server with APC
├── phase1_experiment.py      # Phase 1 experiment → results_phase1.json
├── phase2_experiment.py      # Phase 2 experiment → results_phase2.json
├── plot_results.py           # Reads JSON results, produces PNG plots
├── results_phase1.json       # Phase 1 raw data + statistics
├── results_phase2.json       # Phase 2 raw data + statistics
├── experiment1_results.png   # 2×2 grid: TTFT + throughput per phase
└── experiment1_phase_comparison.png  # Grouped bars across all conditions
```

## Running the Experiments

### Prerequisites

```bash
# Install dependencies (from this directory)
uv sync
```

### Phase 1: HTTP API

```bash
# Terminal 1 — start the vLLM server
bash phase1_server.sh

# Terminal 2 — once the server is ready, run the experiment
uv run python phase1_experiment.py

# Stop the server when done (Ctrl+C in Terminal 1)
```

### Phase 2: Python API

```bash
# No server needed — runs standalone
uv run python phase2_experiment.py
```

### Generate Plots

```bash
# Reads results_phase1.json and results_phase2.json
uv run python plot_results.py
```

## Output Schema

Both phases produce JSON with the same structure:

```json
{
  "phase": "phase1_http_api | phase2_python_api",
  "model": "facebook/opt-125m",
  "timestamp": "...",
  "config": {
    "context_size_a": 512,
    "context_size_b": 256,
    "max_output_tokens": 50,
    "num_trials": 10,
    "warmup_trials": 1
  },
  "cache_hit": {
    "trials": [{ "ttft_ms": ..., "total_time_ms": ..., "throughput": ..., "prompt_tokens": ..., "completion_tokens": ... }],
    "stats": { "ttft_mean": ..., "ttft_median": ..., "ttft_stdev": ..., "throughput_mean": ..., "throughput_stdev": ..., "prompt_tokens": ... }
  },
  "cache_miss": { "trials": [...], "stats": {...} },
  "ttft_degradation_pct": ...
}
```

## Plots

**`experiment1_results.png`** — 2×2 grid showing TTFT and throughput for each phase separately, with error bars (stdev) and degradation % annotations.

**`experiment1_phase_comparison.png`** — Side-by-side grouped bars for all 4 conditions (P1-hit, P1-miss, P2-hit, P2-miss), highlighting HTTP overhead between phases.

## Server Configuration

The vLLM server is launched with conservative settings for 4 GB VRAM:

- `--gpu-memory-utilization 0.7` — leaves headroom for CUDA overhead
- `--max-model-len 1024` — limits KV cache allocation
- `--enforce-eager` — disables CUDA graphs to save VRAM
- `--swap-space 0` — no CPU swap (unnecessary for this model size)
- Custom chat template concatenates message contents (OPT-125m has no native chat format)
