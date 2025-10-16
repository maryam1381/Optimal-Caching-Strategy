import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict, Union

from src.losses import project_capped_simplex
from src.utils import get_device

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
    Evaluates the model, returns performance metrics and inference times.
    Now, inference times are added directly to the metrics dictionary.
    """
    device = get_device()
    model.to(device)
    policy_results = []
    inference_times = []
    pool_size = len(env_pool_list)
    model.eval()

    with torch.no_grad():
        for q_np in val_dataset_q:
            q = torch.from_numpy(q_np).float().to(device)
            
            # Measure inference time
            start_time = time.perf_counter()
            x_raw = model(q.unsqueeze(0)).squeeze(0)
            x = project_fn(x_raw.unsqueeze(0), S).squeeze(0)
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)

            inds = rng.integers(0, pool_size, size=n_env_eval)
            u_vals = []
            for s in inds:
                p_s, lam_s = env_pool_list[int(s)]
                p_t = torch.from_numpy(np.asarray(p_s, dtype=np.float32)).to(device)
                lam_t = torch.tensor(float(lam_s), dtype=torch.float32, device=device)
                u = compute_utility_fn(p_t, lam_t, x)
                u_vals.append(u.item())

            u_arr = np.array(u_vals)
            per_sample_loss = 1 - u_arr
            var_gamma = np.quantile(per_sample_loss, 1 - gamma)
            cvar_gamma = np.mean(per_sample_loss[per_sample_loss >= var_gamma])
            policy_results.append((u_arr.mean(), per_sample_loss.mean(), var_gamma, cvar_gamma))

    arr = np.array(policy_results)
    metrics = {
        'mean_utility_mean': arr[:, 0].mean(),
        'mean_utility_std': arr[:, 0].std(),
        'mean_loss_mean': arr[:, 1].mean(),
        f'VaR_{gamma}': arr[:, 2].mean(),
        f'CVaR_{gamma}': arr[:, 3].mean(),
        'inference_time_median': np.median(inference_times),
        'inference_time_95_percentile': np.percentile(inference_times, 95),
    }
    
    return metrics

def evaluate_all_policies(
    policies_to_evaluate: Dict[str, Callable],
    val_dataset_q: np.ndarray,
    env_pool_list: Sequence[Tuple[np.ndarray, float]],
    project_fn: Callable,
    compute_utility_fn: Callable,
    S: float,
    gamma: float,
    device: torch.device,
    rng: np.random.Generator,
    n_env_eval: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Runs a comprehensive evaluation of multiple policies, including timing,
    and returns a dictionary of metrics for each.
    """
    final_metrics = {}

    for policy_name, policy_fn in policies_to_evaluate.items():
        print(f"  -> Evaluating policy: {policy_name}")

        if isinstance(policy_fn, nn.Module):
            # Evaluate the model (inference time and performance metrics)
            metrics = evaluate_model(
                model=policy_fn,
                val_dataset_q=val_dataset_q,
                env_pool_list=env_pool_list,
                project_fn=project_fn,
                compute_utility_fn=compute_utility_fn,
                S=S,
                gamma=gamma,
                device=device,
                n_env_eval=n_env_eval,
                rng=rng
            )
        else:
            policy_results = []
            pool_size = len(env_pool_list)
            inference_times = []

            for q_np in val_dataset_q:
                # Measure inference time
                start_time = time.perf_counter()
                x_np = policy_fn(q_np)
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)

                x = torch.from_numpy(x_np).float().to(device)

                inds = rng.integers(0, pool_size, size=n_env_eval)
                u_vals = []
                with torch.no_grad():
                    for s in inds:
                        p_s, lam_s = env_pool_list[int(s)]
                        p_t = torch.from_numpy(np.asarray(p_s, dtype=np.float32)).to(device)
                        lam_t = torch.tensor(float(lam_s), dtype=torch.float32, device=device)
                        u = compute_utility_fn(p_t, lam_t, x)
                        u_vals.append(u.item())

                u_arr = np.array(u_vals)
                per_sample_loss = 1 - u_arr
                var_gamma = np.quantile(per_sample_loss,1- gamma)
                cvar_gamma = np.mean(per_sample_loss[per_sample_loss >= var_gamma])
                
                policy_results.append((u_arr.mean(), per_sample_loss.mean(), var_gamma, cvar_gamma))

            arr = np.array(policy_results)
            metrics = {
                'mean_utility_mean': arr[:, 0].mean(),
                'mean_utility_std': arr[:, 0].std(),
                'mean_loss_mean': arr[:, 1].mean(),
                f'VaR_{gamma}': arr[:, 2].mean(),
                f'CVaR_{gamma}': arr[:, 3].mean(),
                'inference_time_median': np.median(inference_times),
                'inference_time_95_percentile': np.percentile(inference_times, 95),
            }

        final_metrics[policy_name] = metrics

    return final_metrics


if __name__ == '__main__':
    # --- Mock objects and functions for demonstration ---
    # MOCK SETUP ASSUMES M=10 files and IN_DIM=5 features
    M = 10
    IN_DIM = 5
    
    class MockModel(nn.Module):
        def __init__(self, output_size=M): # Output size must be M=10
            super().__init__()
            self.layer = nn.Linear(IN_DIM, output_size) # Input size is IN_DIM=5

        def forward(self, x):
            return self.layer(x)

    def mock_project_fn(y, S):
        # A simple mock projection (e.g., sigmoid/normalize)
        return torch.sigmoid(y)

    def mock_compute_utility_fn(p_t, lam_t, x):
    # Use a bounded mock: e.g., sigmoid of the sum, or clip the value.
    # The sum of p_t is approx 1.0. The sum of x is max S.
    
    # A safer mock utility (clamped to [0, 1]):
        raw_utility = 0.5 + 0.5 * torch.sum(p_t * x) / (float(M) * 0.5)
        return torch.clamp(raw_utility - lam_t * 0.1, 0.0, 1.0)

    # --- Setup the evaluation ---
    mock_model = MockModel(output_size=M)
    mock_val_dataset = np.random.rand(50, IN_DIM) # 50 samples, IN_DIM=5 features
    mock_env_pool = [(np.random.rand(M), np.random.rand()) for _ in range(5000)] # 5000 scenarios, M=10 files
    mock_S = 5.0
    mock_gamma = 0.05
    mock_device = torch.device("cpu")
    mock_rng = np.random.default_rng()

    # --- Run the evaluate_model demonstration ---
    print("Running evaluate_model demonstration...")
    try:
        final_metrics = evaluate_model(
            model=mock_model,
            val_dataset_q=mock_val_dataset,
            env_pool_list=mock_env_pool,
            project_fn=project_capped_simplex,
            compute_utility_fn=mock_compute_utility_fn,
            S=mock_S,
            gamma=mock_gamma,
            device=mock_device,
            n_env_eval=200,
            rng=mock_rng
        )
        print("\nAggregated Metrics (evaluate_model):")
        for metric, value in final_metrics.items():
            print(f"  - {metric}: {value:.4f}")

    except Exception as e:
        print(f"An error occurred during evaluate_model: {e}")

    # --- Run the evaluate_all_policies demonstration ---
    def mock_baseline_policy(q_np):
        # Baseline policy function must return a policy x (numpy) of size M=10
        # It takes q_np (size IN_DIM=5) as input but must return size M=10.
        # We simulate a simple policy here that ignores q_np input for simplicity.
        fixed_x = np.full((M,), 0.5)
        return np.clip(fixed_x, 0.0, 1.0) 

    policies_to_eval = {
        "Trained Model": mock_model,
        "Baseline Fixed": mock_baseline_policy
    }

    print("\nRunning evaluate_all_policies demonstration...")
    try:
        all_metrics = evaluate_all_policies(
            policies_to_evaluate=policies_to_eval,
            val_dataset_q=mock_val_dataset,
            env_pool_list=mock_env_pool,
            project_fn=project_capped_simplex, # <--- PASSING project_fn HERE
            compute_utility_fn=mock_compute_utility_fn,
            S=mock_S,
            gamma=mock_gamma,
            device=mock_device,
            rng=mock_rng
        )
        print("\nAggregated Metrics (evaluate_all_policies):")
        for policy, metrics in all_metrics.items():
            print(f"[{policy}]")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value:.4f}")
    except Exception as e:
        print(f"An error occurred during evaluate_all_policies: {e}")