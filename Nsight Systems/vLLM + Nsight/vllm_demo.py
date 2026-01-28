import torch
import torch.cuda.nvtx as nvtx
from vllm import LLM, SamplingParams

print(f"using device: {torch.cuda.get_device_name(0)}")

# load model on gpu
nvtx.range_push("model_initialization")
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.5)
nvtx.range_pop()

# sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

# prompts
prompts = [
    "Hello, my name is",
    "The future of AI is",
]

# run inference on gpu
nvtx.range_push("inference")
outputs = llm.generate(prompts, sampling_params)
nvtx.range_pop()

# print results
nvtx.range_push("output_processing")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"prompt: {prompt!r}\ngenerated: {generated_text!r}\n")
nvtx.range_pop()

print("done")
