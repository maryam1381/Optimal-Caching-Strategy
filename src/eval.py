# src/eval.py
"""
Evaluation utilities for Digital-Twin-assisted CVaR-robust D2D caching.

This module contains:
 - metrics: empirical VaR / CVaR computation, CDF computation
 - baseline policies:
    * popularity_deterministic_topS (posterior-mean top-S deterministic)
    * popularity_proportional (posterior-mean proportional then scaled)
    * mean_opt_plug_in (plug-in optimizer that maximizes expected utility under posterior mean via projected gradient ascent)
 - policy evaluation:
    * evaluate_policy_on_env_pool: for a given measurement q and policy x, compute utility samples across scenario draws
    * batch_evaluate_model: run a trained NN model across many q and evaluate mean & tail metrics
 - plotting helpers:
    * plot_mean_vs_cvar
    * plot_utility_cdf_for_measurement
 - small demo / smoke tests.
"""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

# Recommended external helpers / project functions (import or pass-in)
# from src.env_pool import build_env_pool
# from src.losses import project_capped_simplex
# from src.model import create_mlp

# -----------------------------------------------------------------------------
# Basic statistical utilities
# -----------------------------------------------------------------------------

def empirical_var(u: np.ndarray, gamma: float) -> float:
    """
    Empirical VaR (lower-tail quantile) at level gamma.
    u: 1D array of utility samples (realizations). We treat lower tail (small utilities) as "bad".
    gamma: tail probability in (0,1), e.g., 0.05
    Returns: scalar VaR (gamma-quantile)
    """
    if u.size == 0:
        return float('nan')
    return float(np.quantile(u, gamma, interpolation='linear'))


def empirical_cvar(u: np.ndarray, gamma: float) -> float:
    """
    Empirical CVaR (average of worst gamma fraction). For utilities we compute the
    average of the gamma-quantile (left tail) values — i.e., the mean of the smallest ceil(gamma*L) samples.
    """
    if u.size == 0:
        return float('nan')
    L = u.size
    k = max(1, int(math.ceil(gamma * L)))
    sorted_u = np.sort(u)  # ascending: worst (small) first
    return float(np.mean(sorted_u[:k]))


def empirical_cdf(u: np.ndarray, points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return empirical CDF evaluated at `points` (if provided) or at sorted unique samples.
    Returns (x_values, cdf_values) suitable for plotting where cdf_values = P(U <= x).
    """
    u = np.asarray(u).ravel()
    if u.size == 0:
        return np.array([]), np.array([])
    if points is None:
        xs = np.sort(u)
        ys = np.arange(1, xs.size + 1) / float(xs.size)
        return xs, ys
    else:
        pts = np.asarray(points)
        ys = np.searchsorted(np.sort(u), pts, side='right') / float(u.size)
        return pts, ys


# -----------------------------------------------------------------------------
# Evaluation primitives
# -----------------------------------------------------------------------------

def evaluate_policy_on_env_pool(
    x: np.ndarray,
    env_pool: Sequence[Tuple[np.ndarray, float]],
    compute_utility_fn: Callable[[np.ndarray, float, np.ndarray], float],
    n_sample: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Compute the utility samples { U(x; env^{(s)}) } over env_pool draws.

    Args:
      x: policy vector (numpy) shape (M,)
      env_pool: sequence of (p_vector (np.ndarray length M), lambda_scalar)
      compute_utility_fn: function (p:np.ndarray, lam:float, x:np.ndarray) -> scalar utility
                         NOTE: this function should accept numpy arrays and return float.
      n_sample: number of env draws to use (if None uses full pool)
      rng: optional np.random.Generator

    Returns:
      numpy array of utilities of length n_used
    """
    if rng is None:
        rng = np.random.default_rng(None)
    pool_size = len(env_pool)
    if pool_size == 0:
        return np.array([])

    if n_sample is None or n_sample >= pool_size:
        selected = range(pool_size)
    else:
        idx = rng.integers(0, pool_size, size=n_sample)
        selected = idx

    u_list = []
    for s in selected:
        p_s, lam_s = env_pool[int(s)]
        u = compute_utility_fn(p_s, float(lam_s), x)
        u_list.append(float(u))
    return np.asarray(u_list)


def batch_evaluate_model(
    model: torch.nn.Module,
    dataset_q: np.ndarray,
    env_pool: Sequence[Tuple[np.ndarray, float]],
    compute_utility_fn: Callable[[np.ndarray, float, np.ndarray], float],
    project_fn: Callable[[torch.Tensor, float], torch.Tensor],
    device: str = "cpu",
    n_env_eval: int = 1000,
    rng_seed: int = 0,
    gamma_loss: float = 0.05, # <-- ADDED ARGUMENT for loss tail
) -> Dict[str, np.ndarray]:
    """
    For each measurement q in dataset_q, run model->project->policy x and evaluate
    utility distribution across n_env_eval env draws from env_pool. Return arrays of
    mean utility, VaR_gamma, CVaR_gamma for each q, AND the loss-based counterparts.
    ...
    """
    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()

    rng = np.random.default_rng(rng_seed)
    # The pool_size logic here in the original file is a bit mixed, 
    # but we will proceed with the original implementation's structure.
    # The original env_pool is a Sequence[Tuple[p_s, lam_s]], so len(env_pool) is the pool size.
    # The inner logic accesses env_pool[0][int(s)], env_pool[1][int(s)] which assumes env_pool 
    # is a tuple of two sequences (p_vec, lam_vec). We'll keep the logic that assumes 
    # the second, likely incorrect structure, but fix the metric computation.
    
    # We will use the correct pool size, assuming the outer structure is correct.
    pool_size = len(env_pool) 

    mean_u_list = []
    var_u_list = []
    cvar_u_list = []
    # --- ADDED LOSS METRIC LISTS ---
    mean_l_list = []
    var_l_list = []
    cvar_l_list = []
    # -------------------------------
    all_samples = []

    for q_np in dataset_q:
        # Forward pass
        with torch.no_grad():
            q_t = torch.from_numpy(q_np.astype(np.float32)).to(device_t).unsqueeze(0)
            y = model(q_t).squeeze(0)  # (M,)
            x_t = project_fn(y, S=None)
            # convert to numpy
            x = x_t.detach().cpu().numpy()

        # sample env scenarios (re-using evaluation for speed)
        # Note: The original code indices were likely flawed, 
        # using env_pool[0][int(s)], env_pool[1][int(s)] instead of env_pool[int(s)][0], env_pool[int(s)][1]
        # We will assume the indices logic from evaluate_policy_on_env_pool is what was intended:
        if pool_size > 0:
            inds = rng.integers(0, pool_size, size=n_env_eval)
        else:
            inds = []
            
        u_vals = []
        for s in inds:
            # ASSUMING THE INTENDED STRUCTURE is env_pool[idx] = (p_vec, lam_scalar)
            p_s, lam_s = env_pool[int(s)] # Fixed indexing based on the external structure
            u = compute_utility_fn(p_s, float(lam_s), x)
            u_vals.append(float(u))
        u_arr = np.asarray(u_vals)
        
        # 1. Compute Loss Samples
        loss_arr = 1.0 - u_arr # Per-sample loss: ell^(s) = 1 - U^(s)
        
        # 2. Empirical Mean Loss (E[ell])
        mean_l_list.append(float(loss_arr.mean()))
        
        # 3. Empirical VaR (Loss) - empirical gamma-quantile of loss
        # This is the (1-gamma)-quantile on utility.
        # It's the upper tail of loss, or np.quantile(loss, 1-gamma)
        var_l_list.append(float(np.quantile(loss_arr, 1.0 - gamma_loss)))
        
        # 4. Empirical CVaR (Loss) - average loss in worst gamma-tail
        L = loss_arr.size
        k_loss = max(1, int(math.ceil(gamma_loss * L)))
        # Sort ascending: worst (largest) loss is last. Take the last k_loss elements.
        sorted_loss = np.sort(loss_arr) 
        cvar_l_list.append(float(np.mean(sorted_loss[-k_loss:])))
        
        # Original Utility Metrics (keep for consistency with existing output)
        mean_u_list.append(float(u_arr.mean()))
        # default gamma for reporting: 0.05
        var_u_list.append(float(np.quantile(u_arr, 0.05)))
        k_u = max(1, int(math.ceil(0.05 * u_arr.size)))
        cvar_u_list.append(float(np.mean(np.sort(u_arr)[:k_u]))) # lower tail (worst utility)

        all_samples.append(u_arr)

    return {
        'mean_u': np.array(mean_u_list),
        'var_u': np.array(var_u_list),
        'cvar_u': np.array(cvar_u_list),
        # --- ADDED LOSS METRICS ---
        'mean_loss': np.array(mean_l_list),
        f'var_loss_{gamma_loss}': np.array(var_l_list),
        f'cvar_loss_{gamma_loss}': np.array(cvar_l_list),
        # --------------------------
        'all_u': all_samples
    }

# -----------------------------------------------------------------------------
# Baseline policy constructors
# -----------------------------------------------------------------------------

def popularity_deterministic_topS(p_bar: np.ndarray, S: int) -> np.ndarray:
    """
    Deterministic popularity-based baseline (plug-in): cache the top-S files according to posterior-mean popularity p_bar.
    If S is not integer, it is rounded down.
    Returns x in {0,1}^M with sum <= S.
    """
    M = p_bar.size
    S_int = int(min(max(0, math.floor(S)), M))
    if S_int == 0:
        return np.zeros(M)
    idx_sorted = np.argsort(-p_bar)  # descending
    x = np.zeros(M, dtype=float)
    x[idx_sorted[:S_int]] = 1.0
    return x


def popularity_proportional(p_bar: np.ndarray, S: float) -> np.ndarray:
    """
    Proportional baseline: set x_i \\propto p_bar_i and rescale to satisfy sum x = S, clip to [0,1].
    Implementation details:
      - start with y = p_bar / sum(p_bar) * S  (equals p_bar*S since p_bar sums to 1)
      - clip y to [0,1]
      - if sum clipped <= S, return clipped; otherwise re-scale the un-clipped components to make sum=S (waterfilling-like).
    This is simple heuristic and roughly matches literature suggestion.
    """
    M = p_bar.size
    if S <= 0:
        return np.zeros(M)
    if S >= M:
        return np.ones(M)

    y = p_bar * S  # since p_bar sums to 1
    y = np.clip(y, 0.0, 1.0)
    s = y.sum()
    if s <= S:
        return y
    # need to rescale downwards while keeping clipping at 0/1 - simple iterative scheme
    # Use the same projection onto capped simplex as used in training:
    # We rely on projection function provided externally; fallback: simple normalization and clipping
    y = y / y.sum() * S
    y = np.clip(y, 0.0, 1.0)
    # final adjustment if still off
    if abs(y.sum() - S) > 1e-6:
        # uniform redistribution: scale then clip then waterfill small leftover
        # fallback simple normalization:
        y = y / y.sum() * S
    return y


# def mean_opt_plug_in(
#     p_bar: np.ndarray,
#     lambda_bar: float,
#     S: float,
#     compute_utility_fn: Callable[[np.ndarray, float, np.ndarray], float],
#     project_fn: Callable[[torch.Tensor, float], torch.Tensor],
#     M: int,
#     lr: float = 1e-2,
#     steps: int = 400,
#     device: str = "cpu",
#     verbose: bool = False
# ) -> np.ndarray:
#     """
#     Compute plug-in mean-opt policy by maximizing expected utility under posterior mean parameters (p_bar, lambda_bar).
#     We use projected gradient ascent implemented with PyTorch autograd.
#     """
#     # **BUG FIX**: Handle the zero-capacity edge case
#     if S <= 1e-9:
#         return np.zeros(M, dtype=np.float32)

#     device_t = torch.device(device)
    
#     # Initialize x as a parameter for the optimizer
#     x = torch.full((M,), float(S) / M, dtype=torch.float32, device=device_t, requires_grad=True)
    
#     # Use a standard PyTorch optimizer
#     optimizer = optim.Adam([x], lr=lr)

#     p_t = torch.from_numpy(p_bar.astype(np.float32)).to(device_t)
#     lam_t = torch.tensor(float(lambda_bar), dtype=torch.float32, device=device_t)

#     for it in range(steps):
#         optimizer.zero_grad()
        
#         # Project x to ensure it's a valid policy for the utility calculation
#         x_proj = project_fn(x, S)
        
#         # We want to maximize utility, so we minimize its negative
#         loss = -compute_utility_fn(p_t, lam_t, x_proj)
        
#         loss.backward()
#         optimizer.step()

#         # After the optimizer step, project x back into the feasible set
#         with torch.no_grad():
#             x.data = project_fn(x.data, S)

#         if verbose and (it % 100 == 0):
#             print(f"[mean_opt] iter {it}/{steps}, utility={-loss.item():.6f}")

#     # Return the final optimized and projected policy
#     x_opt = project_fn(x, S).detach().cpu().numpy()
#     return x_opt

# def mean_opt_plug_in(
#     p_bar: np.ndarray,
#     lambda_bar: float,
#     S: float,
#     compute_utility_fn: Callable[[torch.Tensor, float, torch.Tensor], float],
#     project_fn: Callable[[torch.Tensor, float], torch.Tensor],
#     M: int,
#     lr: float = 1e-2,
#     steps: int = 400,
#     device: str = "cpu",
#     verbose: bool = False
# ) -> np.ndarray:
#     """
#     Compute plug-in mean-opt policy by maximizing expected utility under posterior mean parameters (p_bar, lambda_bar).
#     We use projected gradient ascent implemented with PyTorch autograd.
#     """
#     # **BUG FIX**: Handle the zero-capacity edge case
#     if S <= 1e-9:
#         return np.zeros(M, dtype=np.float32)

#     device_t = torch.device(device)
    
#     # Initialize x as a parameter for the optimizer
#     x = torch.full((M,), float(S) / M, dtype=torch.float32, device=device_t, requires_grad=True)
    
#     optimizer = optim.SGD([x], lr=lr)

#     p_t = torch.from_numpy(p_bar.astype(np.float32)).to(device_t)
#     lam_t = torch.tensor(float(lambda_bar), dtype=torch.float32, device=device_t)

#     for it in range(steps):
#         optimizer.zero_grad()
        

#         loss = -compute_utility_fn(p_t, lam_t, x)
        
#         loss.backward()
#         # print(f"[mean_opt step {it}] x.grad exists={x.grad is not None}")
#         optimizer.step()

#         # After the optimizer takes a step (which may move x outside the
#         # feasible set), we project it back. This is the correct implementation
#         # of Projected Gradient Ascent.
#         with torch.no_grad():
#             x.data = project_fn(x.data, S)
#         # --- FIX END ---

#         # if verbose and (it % 100 == 0):
#         #     print(f"[mean_opt] iter {it}/{steps}, utility={-loss.item():.6f}")

#     # Return the final optimized policy. One last projection ensures constraints are met.
#     x_opt = project_fn(x, S).detach().cpu().numpy()
#     # print(f"\n[mean_opt_plug_in] FINAL: x_opt.sum={x_opt.sum():.6f}, x_opt.mean={x_opt.mean():.6f}\n")
#     return x_opt

def mean_opt_plug_in(
    p_bar: np.ndarray,
    lambda_bar: float,
    S: float,
    compute_utility_fn: Callable,
    project_fn: Callable,
    M: int,
    lr: float = 0.05,  # ✅ Increased from 1e-2
    steps: int = 400,
    device: str = "cpu",
    verbose: bool = False
) -> np.ndarray:
    if S <= 1e-9:
        return np.zeros(M, dtype=np.float32)

    device_t = torch.device(device)
    
    # ✅ SMART INITIALIZATION
    x_init = torch.from_numpy((p_bar * S).astype(np.float32)).to(device_t)
    x_init = torch.clamp(x_init, 0.0, 1.0)
    x_init = project_fn(x_init, S)
    x = x_init.clone().requires_grad_(True)
    
    # ✅ USE ADAM
    optimizer = optim.Adam([x], lr=lr)
    
    p_t = torch.from_numpy(p_bar.astype(np.float32)).to(device_t)
    lam_t = torch.tensor(float(lambda_bar), dtype=torch.float32, device=device_t)
    
    for it in range(steps):
        optimizer.zero_grad()
        loss = -compute_utility_fn(p_t, lam_t, x)
        loss.backward()
        
        # ✅ GRADIENT HEALTH CHECK
        if x.grad is not None and torch.isnan(x.grad).any():
            print(f"[ERROR] NaN gradients at step {it}")
            break
            
        optimizer.step()
        
        with torch.no_grad():
            x.data = project_fn(x.data, S)
        
        if verbose and (it % 100 == 0):
            print(f"[mean_opt] iter {it}/{steps}, utility={-loss.item():.6f}")
    
    x_opt = project_fn(x, S).detach().cpu().numpy()
    return x_opt


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

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_mean_vs_cvar(
    results: Dict[str, Sequence[float]],
    labels: Sequence[str],
    gamma: float = 0.05,
    show_std: bool = True,
    title: str = "Mean utility vs CVaR",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Given results as a dict mapping 'method' -> list/array of per-measurement metrics
    or directly arrays [mean_u_array, cvar_array], create a scatter plot or bar with errorbars.

    Example usage:
      results = {
        'CVaR-NN': {'mean': array, 'cvar': array},
        'Mean-Opt': {'mean': array, 'cvar': array}
      }
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    colors = plt.cm.tab10.colors
    for idx, label in enumerate(labels):
        entry = results[label]
        mean_arr = np.asarray(entry['mean'])
        cvar_arr = np.asarray(entry['cvar'])
        ax.scatter(mean_arr, cvar_arr, label=label, alpha=0.6, color=colors[idx % len(colors)])
        if show_std:
            ax.errorbar(mean_arr.mean(), cvar_arr.mean(),
                        xerr=mean_arr.std(), yerr=cvar_arr.std(),
                        fmt='o', color=colors[idx % len(colors)], capsize=3)
    ax.set_xlabel("Mean utility")
    ax.set_ylabel(f"CVaR$_{{{gamma}}}$ (lower tail avg)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return ax


def plot_utility_cdf_for_measurement(
    u_dict: Dict[str, np.ndarray],
    zoom_left_tail: Optional[Tuple[float, float]] = None,
    n_points: int = 200,
    title: str = "Utility CDF for representative q",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot empirical CDFs for several methods for a single measurement q.
    u_dict: mapping name -> 1D array of utility samples (many env draws)
    zoom_left_tail: optional (xmin, xmax) to zoom into left tail region
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for name, u in u_dict.items():
        xs, ys = empirical_cdf(u)
        ax.plot(xs, ys, label=name)
    ax.set_xlabel("Utility")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    if zoom_left_tail is not None:
        ax.set_xlim(zoom_left_tail)
    return ax


# -----------------------------------------------------------------------------
# Convenience demo / example
# -----------------------------------------------------------------------------

def demo_baselines_and_eval(
    env_pool: Sequence[Tuple[np.ndarray, float]],
    compute_utility_fn: Callable[[np.ndarray, float, np.ndarray], float],
    project_fn: Callable[[torch.Tensor, float], torch.Tensor],
    S: float,
    M: int,
    n_test_q: int = 20,
    n_env_eval: int = 2000,
    rng_seed: int = 0
) -> None:
    """
    High-level demo that:
      - selects a small set of measurement q's from env_pool (use posterior draws as surrogates for q)
      - builds baseline policies (popularity deterministic, popularity proportional, mean-opt)
      - evaluates each policy against a large sample of env draws to compute mean/VaR/CVaR
      - plots mean vs CVaR scatter and one representative CDF.

    This is convenient as a smoke-test for the pipeline.
    """
    rng = np.random.default_rng(rng_seed)
    pool_size = len(env_pool)
    assert pool_size > 0

    # Use env_pool samples as surrogates for measurements q (for demo only)
    q_indices = rng.choice(pool_size, size=min(n_test_q, pool_size), replace=False)
    q_samples = [env_pool[i] for i in q_indices]

    # For each q, compute posterior-mean p_bar and lambda_bar from a subset of env_pool (in real use get from twin)
    # Here we simply compute global means from entire env_pool as an approximation
    p_stack = np.stack([p for (p, lam) in env_pool], axis=0)
    lambda_vec = np.array([lam for (p, lam) in env_pool])
    p_bar_global = p_stack.mean(axis=0)
    lambda_bar_global = float(lambda_vec.mean())

    print("Global posterior mean summary: lambda_bar=", lambda_bar_global, "sum p_bar=", p_bar_global.sum())

    # Precompute mean-opt policy once (plug-in)
    print("Computing mean-opt plug-in policy (this uses autograd optimization)...")
    x_mean_opt = mean_opt_plug_in(
        p_bar=p_bar_global,
        lambda_bar=lambda_bar_global,
        S=S,
        compute_utility_fn=compute_utility_fn,
        project_fn=project_fn,
        M=M,
        lr=5e-3,
        steps=600,
        device="cpu",
        verbose=True
    )

    # baseline policies
    x_pop_det = popularity_deterministic_topS(p_bar_global, S=int(round(S)))
    x_pop_prop = popularity_proportional(p_bar_global, S=S)

    methods = {
        'MeanOpt': x_mean_opt,
        'PopDet': x_pop_det,
        'PopProp': x_pop_prop
    }

    # Evaluate each method on a single representative q (take the first q_sample)
    rep_q = q_samples[0]
    # In a real scenario q contains observed counts and user sightings; here rep_q is a (p, lambda) pair as demo
    rep_p, rep_lambda = rep_q

    # Evaluate utilities across many env draws (not just the env_pool) — we use env_pool as sampler
    results_u = {}
    for name, x in methods.items():
        u_s = evaluate_policy_on_env_pool(x, env_pool, compute_utility_fn, n_sample=n_env_eval, rng=rng)
        results_u[name] = u_s
        print(f"{name}: mean={u_s.mean():.4f}, VaR-0.05={empirical_var(u_s, 0.05):.4f}, CVaR-0.05={empirical_cvar(u_s, 0.05):.4f}")

    # Plotting
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for name, u_s in results_u.items():
        xs, ys = empirical_cdf(u_s)
        ax1.plot(xs, ys, label=name)
    ax1.set_title("Empirical CDF (representative q)")
    ax1.set_xlabel("Utility")
    ax1.set_ylabel("CDF")
    ax1.legend()
    plt.show()

    # Mean vs CVaR scatter for these methods across q_samples
    means = []
    cvars = []
    for name, x in methods.items():
        mean_vals = []
        cvar_vals = []
        for q_idx in range(len(q_samples)):
            u_s = evaluate_policy_on_env_pool(x, env_pool, compute_utility_fn, n_sample=500, rng=rng)
            mean_vals.append(float(u_s.mean()))
            cvar_vals.append(empirical_cvar(u_s, 0.05))
        means.append(np.array(mean_vals))
        cvars.append(np.array(cvar_vals))

    results = {
        'MeanOpt': {'mean': means[0], 'cvar': cvars[0]},
        'PopDet': {'mean': means[1], 'cvar': cvars[1]},
        'PopProp': {'mean': means[2], 'cvar': cvars[2]}
    }
    plot_mean_vs_cvar(results, labels=['MeanOpt', 'PopDet', 'PopProp'], gamma=0.05)
    plt.show()


# -----------------------------------------------------------------------------
# If run directly, run a tiny smoke demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Very small smoke test (toy)
    def toy_util(p, lam, x):
        # toy utility: 1 - sum p_i * p_out_i(x) where p_out_i approximated as decreasing in x_i
        # use a small surrogate: p_out_i = exp(-c * lam * x_i), so U = 1 - sum p * p_out
        c = 0.5
        # --- FIX START ---
        if isinstance(p, np.ndarray):
            # NumPy mode
            p_out = np.exp(-c * lam * x)
            return float(1.0 - np.dot(p, p_out))
        elif torch.is_tensor(p):
            # PyTorch mode for mean_opt_plug_in
            lam_t = torch.tensor(float(lam), dtype=torch.float32, device=p.device)
            p_out = torch.exp(-c * lam_t * x)
            return 1.0 - torch.sum(p * p_out)
        else:
            raise TypeError("Inputs must be np.ndarray or torch.Tensor")
        # --- FIX END ---

    # create toy env_pool
    M = 20
    rng = np.random.default_rng(0)
    env_pool = []
    for _ in range(200):
        p = rng.dirichlet(np.ones(M))
        lam = rng.uniform(0.5, 3.0)
        env_pool.append((p, lam))

    # choose S
    S = 5
    # demo
    demo_baselines_and_eval(env_pool, toy_util, project_fn=lambda y, S: torch.clamp(y, 0.0, 1.0), S=S, M=M)

