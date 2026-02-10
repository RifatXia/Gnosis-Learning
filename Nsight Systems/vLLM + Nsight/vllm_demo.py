import torch
import torch.cuda.nvtx as nvtx
from vllm import LLM, SamplingParams
import time
import json
from datetime import datetime
import os

# configuration
model_name = "gpt2-large"
gpu_memory_utilization = 1.0

print(f"using device: {torch.cuda.get_device_name(0)}")

# generate unique profile filename based on model name
profile_name = f"profile_{model_name.replace('/', '_')}"

# initialize benchmark metrics
benchmark = {
    "model_name": model_name,
    "timestamp": datetime.now().isoformat(),
    "model_properties": {},
    "inference_metrics": {},
    "gpu_metrics": {},
    "nsight_metrics": {
        "profile_file": f"{profile_name}.nsys-rep",
        "profile_name": profile_name,
        "notes": "M2D (host-to-device) transfers occur during model loading. D2D (device-to-device) and kernel execution during inference. KV cache usage tracked via GPU block allocation."
    }
}

# get initial gpu memory state
torch.cuda.reset_peak_memory_stats()
initial_memory = torch.cuda.memory_allocated() / (1024**2)

# load model on gpu
nvtx.range_push("model_initialization")
model_load_start = time.time()
# loads the model into GPU memory
llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization)
model_load_time = time.time() - model_load_start
nvtx.range_pop()

# get model properties from vllm engine
model_config = llm.llm_engine.model_config

# calculate total parameters
try:
    total_params = sum(p.numel() for p in llm.llm_engine.model_executor.driver_worker.model_runner.model.parameters())
except:
    # fallback to config if model parameters not accessible
    total_params = getattr(model_config.hf_config, 'num_parameters', None)
    if total_params is None:
        # estimate from hidden size and layers for common architectures
        hidden_size = getattr(model_config.hf_config, 'hidden_size', 0)
        num_layers = getattr(model_config.hf_config, 'num_hidden_layers', 0)
        vocab_size = getattr(model_config.hf_config, 'vocab_size', 0)
        if hidden_size and num_layers and vocab_size:
            # rough estimate: embedding + layers + output
            total_params = vocab_size * hidden_size + num_layers * (12 * hidden_size * hidden_size) + vocab_size * hidden_size
        else:
            total_params = 'N/A'

benchmark["model_properties"] = {
    "total_parameters": total_params if isinstance(total_params, str) else int(total_params),
    "num_layers": getattr(model_config.hf_config, 'num_hidden_layers', 'N/A'),
    "hidden_size": getattr(model_config.hf_config, 'hidden_size', 'N/A'),
    "vocab_size": getattr(model_config.hf_config, 'vocab_size', 'N/A'),
    "model_size_mb": (torch.cuda.memory_allocated() - initial_memory * 1024**2) / (1024**2)
}

# sampling parameters
# temperature controls the creativity/randomness
# top_p nucleus sampling
# max_tokens: max number of tokens per prompt
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

# prompts
prompts = [
    "Hello, my name is",
    "The future of AI is",
]

# run inference on gpu
nvtx.range_push("inference")
inference_start = time.time()
# run the inference using the prompts and sampling_params
# returns generated text and metadata
outputs = llm.generate(prompts, sampling_params)
inference_time = time.time() - inference_start
nvtx.range_pop()

# calculate inference metrics
total_input_tokens = sum(len(llm.llm_engine.tokenizer.encode(p)) for p in prompts)
total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
total_tokens = total_input_tokens + total_output_tokens

benchmark["inference_metrics"] = {
    "model_load_time_seconds": round(model_load_time, 4),
    "total_inference_time_seconds": round(inference_time, 4),
    "tokens_per_second": round(total_output_tokens / inference_time, 2) if inference_time > 0 else 0,
    "num_prompts": len(prompts),
    "total_input_tokens": total_input_tokens,
    "total_output_tokens": total_output_tokens,
    "average_tokens_per_prompt": round(total_output_tokens / len(prompts), 2)
}

# get gpu memory metrics
gpu_props = torch.cuda.get_device_properties(0)
benchmark["gpu_metrics"] = {
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_memory_total_mb": round(gpu_props.total_memory / (1024**2), 2),
    "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
    "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
    "gpu_memory_utilization_percent": round((torch.cuda.memory_allocated() / gpu_props.total_memory) * 100, 2),
    "peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2)
}

# print results
nvtx.range_push("output_processing")
print("\n" + "="*60)
print("INFERENCE RESULTS")
print("="*60)
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nprompt {i+1}: {prompt!r}")
    print(f"generated: {generated_text!r}")

print("\n" + "="*60)
print("BENCHMARK METRICS")
print("="*60)
print(f"Model: {model_name}")
print(f"Model Load Time: {benchmark['inference_metrics']['model_load_time_seconds']:.4f}s")
print(f"Inference Time: {benchmark['inference_metrics']['total_inference_time_seconds']:.4f}s")
print(f"Tokens/Second: {benchmark['inference_metrics']['tokens_per_second']:.2f}")
print(f"GPU Memory Used: {benchmark['gpu_metrics']['gpu_memory_allocated_mb']:.2f} MB ({benchmark['gpu_metrics']['gpu_memory_utilization_percent']:.2f}%)")
print("="*60)
nvtx.range_pop()

# save benchmark to json
output_file = f"benchmark_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(benchmark, f, indent=2)

print(f"\nBenchmark saved to: {output_file}")
print(f"\nTo profile this model with Nsight Systems, run:")
print(f"nsys profile -o {profile_name} --force-overwrite true python3 vllm_demo.py")
print("done")
