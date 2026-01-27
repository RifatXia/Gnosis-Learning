import torch
import torch.cuda.nvtx as nvtx

device = "cuda"
N = 4096

print(f"Using device: {torch.cuda.get_device_name(0)}")
print(f"Matrix size: {N}x{N}")

nvtx.range_push("Data Initialization")
a = torch.randn((N, N), device=device)
b = torch.randn((N, N), device=device)
torch.cuda.synchronize()
nvtx.range_pop()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

nvtx.range_push("Matrix Multiplication Loop")
start_event.record()

for i in range(50):
    nvtx.range_push(f"Iteration {i}")
    c = a @ b
    torch.cuda.synchronize()
    nvtx.range_pop()

end_event.record()
torch.cuda.synchronize()
nvtx.range_pop()

elapsed_time = start_event.elapsed_time(end_event)
print(f"Total time for 50 iterations: {elapsed_time:.2f} ms")
print(f"Average time per iteration: {elapsed_time/50:.2f} ms")
print(f"Done, c[0,0] = {float(c[0,0]):.4f}")
