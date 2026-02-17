import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict, Union

from src.losses import project_capped_simplex
from src.utils import to_torch

# --- Tail metrics for loss right-tail (alpha=0.95) ---
def loss_tail_metrics(losses: np.ndarray, alpha: float = 0.95) -> dict:
    """
    Compute right-tail risk metrics for losses at confidence alpha:
      VaR_loss_alpha: alpha-quantile of loss (e.g., 95th percentile)
      CVaR_loss_alpha: mean of the worst (1 - alpha) fraction of losses
    """
    L = np.asarray(losses, dtype=float)
    var = float(np.quantile(L, alpha))
    tail = L[L >= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    # Monotonicity: CVaR should be >= VaR for losses
    assert cvar >= var - 1e-12
    return {
        f"VaR_loss_{alpha:.2f}": var,
        f"CVaR_loss_{alpha:.2f}": cvar,
    }


def evaluate_model(
    model: nn.Module,
    val_dataset_q: np.ndarray,
    env_pool_list: Sequence[Tuple[np.ndarray, float]],
    project_fn: Callable,
    compute_utility_fn: Callable,
    S: float,
    gamma: float,
    device: torch.device,
    rng: np.random.Generator,
    n_env_eval: int = 500
) -> Dict[str, float]:
    """
    Evaluate the trained model under sampled posterior environments.

    Args:
        model: trained caching policy network
        val_dataset_q: validation measurement vectors, shape (N, in_dim)
        env_pool_list: list of posterior samples [(p_s, λ_s)]
        project_fn: projection function to feasible caching region
        compute_utility_fn: callable(p_t, λ_t, x) -> scalar utility in [0,1]
        S: caching capacity (projection sum constraint)
        gamma: confidence level for CVaR (e.g., 0.95)
        device: torch device
        rng: numpy Generator
        n_env_eval: number of sampled environments per q

    Returns:
        dict of aggregated metrics (mean utility, CVaR, inference times)
    """

    model.eval()
    pool_size = len(env_pool_list)
    inference_times, mean_utils, losses = [], [], []

    with torch.no_grad():
        for q_np in val_dataset_q:
            q = to_torch(q_np).to(device).unsqueeze(0)

            # --- forward + projection ---
            t0 = time.perf_counter()
            y = model(q)
            x = project_fn(y, S)
            t1 = time.perf_counter()
            inference_times.append(t1 - t0)

            # --- sample environment indices ---
            idx = rng.integers(0, pool_size, size=n_env_eval)
            
            p_batch = torch.tensor(
                np.stack([env_pool_list[i][0] for i in idx]), 
                dtype=torch.float32, 
                device=device
            ) # shape (n_env_eval, M)
            
            lam_batch = torch.tensor(
                [env_pool_list[i][1] for i in idx], 
                dtype=torch.float32, 
                device=device
            ) # shape (n_env_eval,)

            # --- vectorized utility computation ---
            # محاسبه یوتیلیتی برای هر محیط نمونه‌برداری شده با توجه به تصمیم x
            u_vals = torch.stack([
                compute_utility_fn(p_batch[j], lam_batch[j], x.squeeze(0))
                for j in range(n_env_eval)
            ])
            u_arr = u_vals.cpu().numpy()

            # --- compute losses and metrics ---
            per_loss = 1.0 - u_arr
            mean_utils.append(u_arr.mean())
            losses.extend(per_loss)

    # --- aggregate across all validation samples ---
    L = np.array(losses)
    var_gamma = np.quantile(L, gamma)        # gamma = 0.95 means 95th percentile
    cvar_gamma = L[L >= var_gamma].mean()    # worst 5%

    metrics = {
        "mean_utility": float(np.mean(mean_utils)),
        "mean_loss_mean": 1.0 - float(np.mean(mean_utils)),
        "utility_std": float(np.std(mean_utils)),
        f"VaR_loss_{gamma:.2f}": float(var_gamma),
        f"CVaR_loss_{gamma:.2f}": float(cvar_gamma),
        "median_inference_time": float(np.median(inference_times)),
        "p95_inference_time": float(np.percentile(inference_times, 95)),
    }
    
    return metrics