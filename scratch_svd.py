import torch
import time

x = torch.randn(4096, 1536, device='cuda')
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    x_c = x - x.mean(dim=0)
    s = torch.linalg.svdvals(x_c)
torch.cuda.synchronize()
print(f"Time per SVD: {(time.time() - start) / 100 * 1000:.2f} ms")
