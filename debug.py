from src.model import create_mlp
import torch

M = 50
model, _ = create_mlp(M, in_extra=1)
q = torch.rand(8, M + 1)
y = model(q)
print("y shape:", y.shape)        # (8, M)
print("y mean/std:", y.mean().item(), y.std().item())

from src.losses import project_capped_simplex
x = project_capped_simplex(y, S=10.0)
print("x sum:", x.sum(dim=1))