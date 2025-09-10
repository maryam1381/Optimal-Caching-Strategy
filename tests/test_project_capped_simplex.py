import torch
from src.losses import project_capped_simplex


def naive_projection(y: torch.Tensor, S: float, tol: float = 1e-6, max_iter: int = 60) -> torch.Tensor:
    """Reference projection using scalar bisection (slow)."""
    y_clamped = y.clamp(0.0, 1.0)
    if y_clamped.sum() <= S:
        return y_clamped
    theta_lo = (y_clamped - 1.0).min()
    theta_hi = y_clamped.max()
    for _ in range(max_iter):
        theta = 0.5 * (theta_lo + theta_hi)
        candidate = torch.clamp(y_clamped - theta, 0.0, 1.0)
        if candidate.sum() > S:
            theta_lo = theta
        else:
            theta_hi = theta
        if (theta_hi - theta_lo).abs() < tol:
            break
    theta = 0.5 * (theta_lo + theta_hi)
    return torch.clamp(y_clamped - theta, 0.0, 1.0)


def test_projection_matches_naive():
    torch.manual_seed(0)
    y = torch.randn(5, 7)
    S = 3.0
    x = project_capped_simplex(y, S)
    x_ref = torch.stack([naive_projection(row, S) for row in y], dim=0)
    assert torch.allclose(x, x_ref, atol=1e-6)


def test_gradient_flow():
    torch.manual_seed(0)
    y = torch.randn(4, dtype=torch.double, requires_grad=True)
    S = 2.0

    def func(inp):
        return project_capped_simplex(inp, S)

    torch.autograd.gradcheck(func, (y,), eps=1e-6, atol=1e-4)

