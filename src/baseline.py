# src/baselines.py
"""
Deterministic policy baselines (Plug-in Mean Opt and Popularity Heuristics)
and utility/risk helper functions for evaluation.
"""
from typing import Sequence, Tuple, Callable
import math
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

# ----------------------------
# Risk Calculation Helpers (Required by demo_baselines_and_eval)
# ----------------------------

def empirical_var(u_s: np.ndarray, gamma: float) -> float:
    """Empirical VaR (Value at Risk) on utility (worst-case utility)."""
    # VaR on utility is the gamma-quantile (low quantile)
    # np.quantile uses ascending order, so gamma-quantile finds the utility
    # level U_gamma such that P(U <= U_gamma) = gamma
    return float(np.quantile(u_s, gamma))

def empirical_cvar(u_s: np.ndarray, gamma: float) -> float:
    """Empirical CVaR (Conditional Value at Risk) on utility."""
    # CVaR on utility is the mean of the worst gamma fraction of utilities.
    # The worst is the lowest gamma fraction.
    sorted_u = np.sort(u_s)
    k = max(1, int(np.ceil(gamma * len(sorted_u))))
    # Take the mean of the k smallest utilities
    return float(np.mean(sorted_u[:k]))

def empirical_cdf(u_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes x and y vectors for plotting the empirical CDF."""
    u_sorted = np.sort(u_s)
    y_cdf = np.arange(1, len(u_sorted) + 1) / len(u_sorted)
    return u_sorted, y_cdf

# ----------------------------
# Utility Evaluation Wrapper (Required by demo_baselines_and_eval)
# ----------------------------

def evaluate_policy_on_env_pool(
    x: np.ndarray,
    env_pool: Sequence[Tuple[np.ndarray, float]],
    compute_utility_fn: Callable,
    n_sample: int,
    rng: np.random.Generator,
    device: str = "cpu"
) -> np.ndarray:
    """
    Evaluates a fixed policy vector x against a large set of scenario draws
    from the env_pool.

    Returns:
        np.ndarray: Array of realized utility scores (shape n_sample,).
    """
    device = torch.device(device)
    pool_size = len(env_pool)

    # Convert fixed policy x to torch tensor
    x_t = torch.from_numpy(x).float().to(device)

    # Sample n_sample scenarios
    inds = rng.integers(0, pool_size, size=n_sample)

    u_vals = []
    with torch.no_grad():
        for s in inds:
            p_s, lam_s = env_pool[int(s)]
            p_t = torch.from_numpy(np.asarray(p_s, dtype=np.float32)).to(device)
            lam_t = torch.tensor(float(lam_s), dtype=torch.float32, device=device)
            u = compute_utility_fn(p_t, lam_t, x_t) # Ensure compute_utility_fn handles scalar p_t, lam_t
            u_vals.append(u.item())

    return np.array(u_vals)


# ----------------------------
# 1. Popularity Heuristics
# ----------------------------

def popularity_deterministic_topS(p_bar: np.ndarray, S: int) -> np.ndarray:
    """
    Popularity heuristic: Deterministic Top-S caching policy.
    Cache the S files with the highest mean popularity (p_bar).
    x_i is 1 for top S, 0 otherwise.

    Args:
        p_bar (np.ndarray): Mean popularity vector (M,).
        S (int): Cache capacity (must be integer for deterministic).

    Returns:
        np.ndarray: Caching marginal vector x (M,).
    """
    x = np.zeros_like(p_bar)
    if S == 0:
        return x

    # Find indices of top S popular items
    top_s_indices = np.argsort(p_bar)[-S:]
    x[top_s_indices] = 1.0
    return x

def popularity_proportional(p_bar: np.ndarray, S: float) -> np.ndarray:
    """
    Popularity heuristic: Proportional caching policy.
    x_i = S * p_i / sum(p_j).
    Since p_bar sums to 1 (it's a probability vector), x_i = S * p_i.
    Capped at 1.

    Args:
        p_bar (np.ndarray): Mean popularity vector (M,).
        S (float): Cache capacity.

    Returns:
        np.ndarray: Caching marginal vector x (M,).
    """
    # Assuming p_bar sums to 1: sum(x) = S * sum(p_bar) = S.
    x = S * p_bar
    # Capping at 1 (individual constraint)
    return np.clip(x, 0.0, 1.0)


# ----------------------------
# 2. Plug-in Mean Optimal Policy (Convex Surrogate)
# ----------------------------

def mean_opt_plug_in(
    p_bar: np.ndarray,
    lambda_bar: float,
    S: float,
    compute_utility_fn: Callable,
    project_fn: Callable,
    M: int,
    lr: float = 5e-3,
    steps: int = 600,
    device: str = "cpu",
    verbose: bool = False
) -> np.ndarray:
    """
    Computes the mean-optimal policy x by minimizing the *negative* mean utility
    (convex surrogate) via gradient descent.

    Uses a plug-in estimate of the environment: E[U(x)] approx U(x; p_bar, lambda_bar).
    The utility function U(x; p, lambda) is assumed to be concave in x, making
    -U convex.

    Args:
        p_bar (np.ndarray): Plug-in mean popularity vector.
        lambda_bar (float): Plug-in mean user density.
        ... other optimization parameters.

    Returns:
        np.ndarray: The optimized caching marginal vector x (M,).
    """
    device = torch.device(device)
    p_t = torch.from_numpy(p_bar).float().to(device)
    lam_t = torch.tensor(lambda_bar, dtype=torch.float32, device=device)

    # Initialize raw scores y (unprojected output)
    # Initial policy x will be uniform: x_init = S/M
    y_init_val = S / M # Simple initialization
    y = torch.full((M,), y_init_val, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = Adam([y], lr=lr)

    if verbose:
        print(f"  Starting mean-opt plug-in optimization (steps={steps}, lr={lr})...")

    best_x = None
    best_utility = -float('inf')

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        # 1. Project y to feasible policy x
        x = project_fn(y, S)

        # 2. Compute mean utility (plug-in estimate)
        # We assume compute_utility_fn handles scalar p, lam input with M-dim x
        # p_t and lam_t should be single tensors (not batches)
        U_mean = compute_utility_fn(p_t, lam_t, x)

        # 3. Objective: Minimize Negative Utility
        loss = -U_mean

        # 4. Backward pass
        loss.backward()

        # 5. Optimization step
        optimizer.step()

        # 6. Tracking best result (optional, but good practice for gradient descent)
        with torch.no_grad():
            if U_mean.item() > best_utility:
                best_utility = U_mean.item()
                best_x = x.cpu().numpy()

        if verbose and step % (steps // 5) == 0:
            print(f"  Step {step:03d}: Loss (Neg Utility)={loss.item():.6f}, "
                  f"Utility={U_mean.item():.6f}")

    return best_x if best_x is not None else x.detach().cpu().numpy()

# ----------------------------
# Policy wrappers used by main.py's `evaluate_all_policies`
# ----------------------------

def plugin_mean_policy(q: np.ndarray, env_pool: Tuple[np.ndarray, np.ndarray], compute_utility_fn: Callable, S: float, project_fn: Callable, M: int, rng: np.random.Generator, device: str) -> np.ndarray:
    """
    Policy function for the Plug-in Mean Optimal Baseline.
    NOTE: In a real system, the posterior means (p_bar, lambda_bar) would be
    calculated based on the measurement q. For a simple baseline, we use the
    *global* posterior mean from the entire env_pool as a plug-in estimate for all q.
    """
    # Calculate global means (p_bar, lambda_bar) from the whole pool for simplicity
    p_pool, lam_pool = env_pool
    p_bar_global = p_pool.mean(axis=0)
    lambda_bar_global = lam_pool.mean()

    # Find the optimal policy for this plug-in estimate
    x_opt = mean_opt_plug_in(
        p_bar=p_bar_global,
        lambda_bar=float(lambda_bar_global),
        S=S,
        compute_utility_fn=compute_utility_fn,
        project_fn=project_fn,
        M=M,
        lr=5e-3, # Use fixed LR for this optimization
        steps=600,
        device=device,
        verbose=False
    )
    return x_opt

def popularity_heuristic(q: np.ndarray, M: int, S: float, device: str) -> np.ndarray:
    """
    Policy function for the Popularity Heuristic Baseline (Top-S based on observed popularity).

    The measurement q is expected to contain the observed request counts (or normalized
    frequencies) which serves as the plug-in popularity estimate p_hat.

    Args:
        q (np.ndarray): The measurement vector (M+1), where q[:M] are normalized popularities.

    Returns:
        np.ndarray: Caching marginal vector x (M,).
    """
    # q[:M] contains the normalized request counts n_norm, which is the popularity estimate p_hat
    p_hat = q[:M]
    # Use the proportional approach, which often performs better than deterministic
    return popularity_proportional(p_hat, S=S)
