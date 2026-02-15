# KV Cache Prefix Invalidation Experiment

Measures the performance cost of KV cache invalidation when context is removed from a multi-turn conversation. Uses llama-cpp-python's direct API with a GGUF model to isolate cache behavior without HTTP overhead.

## Hypothesis

When a shared prefix changes (context A removed), the KV cache cannot reuse previously computed attention states, forcing full recomputation of the remaining context. This should produce measurably higher TTFT compared to a cache-hit scenario where the full prefix is preserved.

## Setup

| Component | Value |
|-----------|-------|
| Model | GPT-2 (GGUF format, ~500 MB) |
| GPU | NVIDIA GeForce GTX 1650 (4 GB VRAM) |
| Runtime | llama-cpp-python (direct API, no HTTP server) |
| Context window | 2048 tokens |
| Output | 50 tokens max per generation |
| Trials | 10 per condition (20 warmups total) |

## Experiment Design

Each trial follows this flow:

```
clear_kv_cache(llm)                     # reset state + clear GPU cache
warmup [A, B] with 6 tokens            # populate KV cache with prefix
sleep 500ms                             # let GPU settle
measure TTFT (1-token generation)       # time to first token
sleep 500ms
measure throughput (50-token generation) # generation speed
```

Two conditions are compared back-to-back:

- **Cache Hit** — query uses full `[A, B, Q]` prefix (matches the warmed cache)
- **Cache Miss** — query uses `[B, Q]` only (A removed, prefix changed)

### Cache Clearing

The KV cache is fully cleared between every trial using two calls:

```python
llm.reset()                # reset n_tokens counter to 0
llm._ctx.kv_cache_clear()  # actually clear KV cache data from GPU
```

`llm.reset()` alone only resets the token counter — it does **not** clear the cached KV data. Both calls are required.

### EOS Handling

GPT-2's EOS token (id 50256) can fire at any point during generation, causing `completion_tokens < max_tokens`. The experiment logs `completion_tokens` and `finish_reason` for every trial and warns if any trial stops early due to EOS.

### Context Size Configurations

Three configurations are tested, keeping A + B = 768 tokens total:

| Config | Context A (removable) | Context B (retained) |
|--------|----------------------|---------------------|
| 1 | 512 tokens | 256 tokens |
| 2 | 384 tokens | 384 tokens |
| 3 | 256 tokens | 512 tokens |

Larger B means more context must be reprocessed on a cache miss, so TTFT degradation scales with B.

## File Structure

```
kv-cache/Test/
├── pyproject.toml              # uv project config + dependencies
├── phase3b_experiment.py       # Main experiment script
├── plot_results.py             # Reads CSV results, produces PNG plots
├── results/                    # Per-trial metrics (CSV)
│   ├── results_phase3b_a512_b256.csv
│   ├── results_phase3b_a384_b384.csv
│   └── results_phase3b_a256_b512.csv
├── prompts/                    # Prompt text, token counts, sample outputs (CSV)
│   ├── prompts_phase3b_a512_b256.csv
│   ├── prompts_phase3b_a384_b384.csv
│   └── prompts_phase3b_a256_b512.csv
└── plots/                      # Generated visualizations
    └── experiment1_all_configs_phase3b.png
```

## Running the Experiment

```bash
# Install dependencies
uv sync

# Run all three configurations
.venv/bin/python phase3b_experiment.py --size-a 512 --size-b 256
.venv/bin/python phase3b_experiment.py --size-a 384 --size-b 384
.venv/bin/python phase3b_experiment.py --size-a 256 --size-b 512

# Generate plots
.venv/bin/python plot_results.py
```

## Output Format

### Results CSV (`results/results_phase3b_a{A}_b{B}.csv`)

20 rows (10 hit + 10 miss), written once at the end with `flush()` + `os.fsync()`:

| Column | Description |
|--------|-------------|
| `trial` | Trial number (1-10) |
| `condition` | `hit` or `miss` |
| `warmup_time_ms` | Time to warm the [A,B] prefix cache |
| `ttft_ms` | Time to first token (1-token generation) |
| `total_time_ms` | Total time for 50-token generation |
| `throughput_tok_s` | Tokens per second during generation |
| `prompt_tokens` | Number of prompt tokens (from tokenizer) |
| `completion_tokens` | Actual tokens generated (may be < 50 if EOS hit) |
| `finish_reason` | `length` (hit max_tokens) or `stop` (EOS hit early) |

### Prompts CSV (`prompts/prompts_phase3b_a{A}_b{B}.csv`)

3 rows (warmup, hit, miss):

| Column | Description |
|--------|-------------|
| `prompt_name` | `warmup`, `hit`, or `miss` |
| `token_count` | Exact token count from `llm.tokenize()` |
| `prompt_text` | Full prompt text sent to the model |
| `sample_output` | Generated text from the first trial |

## Results

### TTFT Degradation by Configuration

| Config (A/B) | Cache Hit TTFT | Cache Miss TTFT | Degradation |
|-------------|---------------|----------------|-------------|
| 512 / 256 | ~28 ms | ~150 ms | +434% |
| 384 / 384 | ~29 ms | ~221 ms | +672% |
| 256 / 512 | ~28 ms | ~281 ms | +906% |

Cache miss TTFT scales linearly with the size of context B (the part that must be reprocessed). Cache hit TTFT stays constant (~28 ms) regardless of configuration since the entire prefix is already cached.

Generation throughput (~210 tok/s) is unaffected by cache state — caching only impacts the prefill phase, not token-by-token decoding.
