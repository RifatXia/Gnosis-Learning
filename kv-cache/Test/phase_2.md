You can the do a phase two directly inside vllm

# Phase 2: vLLM Python API

Same experiment as Phase 1, but using vLLM's `LLM` class directly instead of the HTTP server. This eliminates network overhead and gives tighter measurements.

## What Changes

### Engine initialization (replaces server launch)

Phase 1 required starting a separate server process. Phase 2 instantiates the engine in-process:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,
    max_model_len=32768,
)
```

### Prompt construction (replaces message dicts)

`llm.generate()` takes raw prompt strings, not chat message lists. We apply the model's chat template to get the same tokenized output the server would produce:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

### TTFT measurement (replaces streaming)

No streaming available with the offline API. We approximate TTFT by generating exactly 1 token — the measured time is `prefill + 1 decode step`. The single decode step is constant across conditions, so the T2 vs T3 delta still isolates the cache effect:

```python
params = SamplingParams(max_tokens=1, temperature=0.0)

start = time.perf_counter()
outputs = llm.generate([prompt], params, use_tqdm=False)
ttft_ms = (time.perf_counter() - start) * 1000
```

### Token counts (replaces usage field)

Prompt and output token counts come from the output object directly:

```python
prompt_tokens = len(outputs[0].prompt_token_ids)
output_tokens = len(outputs[0].outputs[0].token_ids)
```

## What Stays the Same

- Experiment design (T1 warmup → T2 cache hit → T3 cache miss)
- Synthetic context generation
- Trial structure (warmup before every trial, discard warmup trials)
- Metrics collected (TTFT, total time, throughput, prompt tokens)
- All parameters (`context_size_a`, `context_size_b`, `num_trials`, etc.)

## Dependencies

Phase 1: `aiohttp`, `numpy`
Phase 2: `vllm`, `transformers`, `numpy`