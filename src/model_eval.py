import math
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Tuple, Callable, Dict, Any

def evaluate_policy_on_env_pool(
    x: np.ndarray,
    env_pool_list: Sequence[Tuple[np.ndarray, float]],
    compute_utility_fn: Callable,
    n_sample: int,
    rng: np.random.Generator,
    device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """
    Evaluates a fixed policy vector x against a large set of scenario draws
    from the env_pool. This is used by the baseline comparison demo.

    Args:
        x (np.ndarray): The fixed policy vector (M,).
        env_pool_list (list): A list of environmental scenarios [(p_s, lam_s), ...].
        compute_utility_fn (function): Utility function.
        n_sample (int): The number of scenario draws to use for evaluation.
        rng (np.random.Generator): NumPy random number generator.
        device (torch.device): The device to run the computation on.

    Returns:
        np.ndarray: Array of realized utility scores (shape n_sample,).
    """
    pool_size = len(env_pool_list)
    
    # Convert fixed policy x to torch tensor
    x_t = torch.from_numpy(x).float().to(device)

    # Sample n_sample scenarios
    inds = rng.integers(0, pool_size, size=n_sample)

    u_vals = []
    with torch.no_grad():
        for s in inds:
            p_s, lam_s = env_pool_list[int(s)]
            p_t = torch.from_numpy(np.asarray(p_s, dtype=np.float32)).to(device)
            lam_t = torch.tensor(float(lam_s), dtype=torch.float32, device=device)
            # compute_utility_fn should handle scalar p_t, lam_t and M-dim x_t
            u = compute_utility_fn(p_t, lam_t, x_t) 
            u_vals.append(u.item())

    return np.array(u_vals)


def evaluate_model(model, val_dataset_q, env_pool_list, project_fn, compute_utility_fn, S,
                   gamma=0.05, device=torch.device("cpu"), n_env_eval=200, rng=np.random.default_rng()):
    """
    Evaluates the trained model (RL2O-CVaR) on the validation set.
    """
    # Check if a model (PyTorch Module) is provided or if the input is a function
    is_model_module = isinstance(model, nn.Module)
    if is_model_module:
        model = model.to(device)
        model.eval()

    results = []
    pool_size = len(env_pool_list)
    for q_np in val_dataset_q:
        x = None
        if is_model_module:
            with torch.no_grad():
                q_t = torch.from_numpy(q_np).float().to(device).unsqueeze(0)
                y = model(q_t).squeeze(0)
                x = project_fn(y, S)
                x_np = x.cpu().numpy()
        else:
            # If a function is passed instead of a model (e.g., a lambda for a policy)
            # the policy function must return the projected policy x (numpy array)
            x_np = model(q_np) 
            x = torch.from_numpy(x_np).float().to(device)

        # Draw n_env_eval scenarios from the environment pool
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
        mean_u = float(u_arr.mean())
        
        # VaR/CVaR calculations on loss
        per_sample_loss = 1 - u_arr
        mean_loss = float(per_sample_loss.mean())

        # VaR on loss: empirical gamma-quantile
        var_gamma = np.quantile(per_sample_loss, gamma)

        # CVaR on loss: mean of the worst gamma fraction of losses.
        sorted_losses = np.sort(per_sample_loss)
        k = max(1, int(np.ceil(gamma * len(sorted_losses))))
        cvar_gamma = np.mean(sorted_losses[-k:])

        results.append((mean_u, mean_loss, var_gamma, cvar_gamma))

    # Aggregate across validation set
    arr = np.array(results)
    metrics = {
        'mean_utility_mean': float(arr[:, 0].mean()),
        'mean_utility_std': float(arr[:, 0].std()),
        'mean_loss_mean': float(arr[:, 1].mean()),
        f'VaR_{gamma}': float(arr[:, 2].mean()),
        f'CVaR_{gamma}': float(arr[:, 3].mean()),
    }
    return metrics


def evaluate_all_policies(
    policies_to_evaluate: Dict[str, Callable],
    val_dataset_q: np.ndarray,
    env_pool_list: Sequence[Tuple[np.ndarray, float]],
    project_fn: Callable, # Added project_fn here
    compute_utility_fn: Callable,
    S: float,
    gamma: float,
    device: torch.device,
    rng: np.random.Generator,
    n_env_eval: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Runs a comprehensive evaluation of multiple policies (trained model and baselines)
    on the validation set and returns a dictionary of metrics for each policy.

    Args:
        policies_to_evaluate (Dict[str, Callable]): Dictionary where keys are policy names
            and values are the policy function (e.g., model or lambda function for baseline).
        project_fn (Callable): The projection function needed by the trained model.
        ... other evaluation parameters ...

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary of metrics per policy.
    """
    final_metrics = {}

    for policy_name, policy_fn in policies_to_evaluate.items():
        print(f"  -> Evaluating policy: {policy_name}")

        # If the policy is a PyTorch Module (your trained model), we use evaluate_model directly.
        if isinstance(policy_fn, nn.Module):
            metrics = evaluate_model(
                model=policy_fn,
                val_dataset_q=val_dataset_q,
                env_pool_list=env_pool_list,
                project_fn=project_fn, # Pass the project_fn from evaluate_all_policies's parameters
                compute_utility_fn=compute_utility_fn,
                S=S,
                gamma=gamma,
                device=device,
                n_env_eval=n_env_eval,
                rng=rng
            )
        
        # If the policy is a function (e.g., a lambda for a baseline), we run the evaluation loop manually
        else:
            policy_results = []
            pool_size = len(env_pool_list)
            
            for q_np in val_dataset_q:
                # Get policy x (numpy array) from the policy function
                x_np = policy_fn(q_np) 
                x = torch.from_numpy(x_np).float().to(device)

                # Draw n_env_eval scenarios from the environment pool
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
                mean_u = float(u_arr.mean())
                
                # VaR/CVaR calculations on loss
                per_sample_loss = 1 - u_arr
                mean_loss = float(per_sample_loss.mean())

                var_gamma = np.quantile(per_sample_loss, gamma)
                sorted_losses = np.sort(per_sample_loss)
                k = max(1, int(np.ceil(gamma * len(sorted_losses))))
                cvar_gamma = np.mean(sorted_losses[-k:])

                policy_results.append((mean_u, mean_loss, var_gamma, cvar_gamma))

            # Aggregate results for this policy
            arr = np.array(policy_results)
            metrics = {
                'mean_utility_mean': float(arr[:, 0].mean()),
                'mean_utility_std': float(arr[:, 0].std()),
                'mean_loss_mean': float(arr[:, 1].mean()),
                f'VaR_{gamma}': float(arr[:, 2].mean()),
                f'CVaR_{gamma}': float(arr[:, 3].mean()),
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
            project_fn=mock_project_fn,
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
            project_fn=mock_project_fn, # <--- PASSING project_fn HERE
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