# src/final_comparison.py
"""
Provides functions to evaluate and compare multiple caching policies
(RL2O-CVaR model, Top-S, and other baselines) on a test dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from numpy.random import Generator
from typing import List, Tuple, Callable, Dict, Any, Sequence

# --- Import necessary functions from your project files ---

# From src/model_eval.py:
from src.model_eval import evaluate_model, loss_tail_metrics

# From src/eval.py:
from src.eval import popularity_deterministic_topS, evaluate_policy_on_env_pool

# From src/losses.py:
from src.losses import single_utility_from_env, project_capped_simplex

# From src/utils.py
from src.utils import set_seed, get_device

# --- Main Comparison Orchestrator ---

def run_final_comparison(
    model: nn.Module,
    test_dataset_q: np.ndarray,
    env_pool_list: List[Tuple[np.ndarray, float]],
    project_fn: Callable,
    S: float,
    gamma: float,
    device: torch.device,
    rng: Generator,
    n_env_eval_model: int = 500,
    n_env_eval_baseline: int = 5000
) -> pd.DataFrame:
    """
    Runs a comparison of all defined models on the test set.
    
    Args:
        model: The trained nn.Module (your RL2O-CVaR model).
        test_dataset_q: The test dataset (numpy array of q observations).
        env_pool_list: A list of (p, lambda) tuples representing environment scenarios.
        project_fn: The projection function (e.g., project_capped_simplex).
        S: The cache capacity (float).
        gamma: The CVaR tail risk level (e.g., 0.05).
        device: The torch.device (cpu or cuda) to run on.
        rng: A numpy.random.Generator for sampling.
        n_env_eval_model: Number of env samples to use *per q* for dynamic models.
        n_env_eval_baseline: Number of env samples to use for *static* baseline policies.

    Returns:
        A pandas DataFrame containing the aggregated performance metrics for all methods.
    """
    
    print("--- Starting Final Model Comparison ---")
    all_results = {}
    alpha = 1.0 - gamma  # Loss tail (e.g., 0.95)

    # --- 0. Setup ---
    # Create the utility function wrapper for numpy-based evaluators
    utility_wrapper = create_numpy_utility_wrapper(device)
    
    # Calculate global p_bar for static policies
    if not env_pool_list:
        raise ValueError("env_pool_list is empty. Cannot compute p_bar.")
    p_all = np.stack([p for p, lam in env_pool_list])
    p_bar_global = p_all.mean(axis=0)

    # --- 1. Evaluate Your Trained Model (RL2O-CVaR) ---
    print(f"[1/3] Evaluating trained model (RL2O-CVaR) on {len(test_dataset_q)} test observations...")
    metrics_model = evaluate_rl2o_cvar_model(
        model=model,
        val_dataset_q=test_dataset_q,
        env_pool_list=env_pool_list,
        project_fn=project_fn,
        S=S,
        gamma=gamma,
        device=device,
        rng=rng,
        n_env_eval=n_env_eval_model,
        alpha=alpha
    )
    all_results["RL2O-CVaR (Model)"] = metrics_model
    print("     ... Trained model evaluation complete.")


    # --- 2. Evaluate the Popularity Top-S Baseline ---
    print(f"[2/3] Evaluating static baseline (Popularity Top-S)...")
    metrics_topS = evaluate_top_s_baseline(
        p_bar_global=p_bar_global,
        S=S,
        env_pool_list=env_pool_list,
        compute_utility_fn=utility_wrapper,
        n_env_eval=n_env_eval_baseline,
        rng=rng,
        alpha=alpha
    )
    all_results["Popularity Top-S"] = metrics_topS
    print("     ... Top-S baseline evaluation complete.")

    
    # # --- 3. Evaluate the New Method (Placeholder) ---
    # print(f"[3/3] Evaluating new method (Placeholder)...")
    # metrics_new = evaluate_new_method_placeholder(
    #     p_bar_global=p_bar_global,
    #     test_dataset_q=test_dataset_q,
    #     S=S,
    #     env_pool_list=env_pool_list,
    #     compute_utility_fn=utility_wrapper,
    #     n_env_eval=n_env_eval_baseline,
    #     rng=rng,
    #     alpha=alpha
    # )
    # all_results["New Method (Placeholder)"] = metrics_new
    # print("     ... New method evaluation complete.")

    
    print("--- Comparison Finished ---")
    
    # Convert final results to a DataFrame for easy viewing
    df_results = pd.DataFrame(all_results).T
    return df_results

# --- Helper Function ---

def create_numpy_utility_wrapper(device: torch.device) -> Callable:
    """
    Creates a wrapper function that allows the numpy-based evaluator
    (evaluate_policy_on_env_pool) to use the PyTorch-based utility function
    (single_utility_from_env).
    """
    def numpy_utility_wrapper(p_np: np.ndarray, lam_float: float, x_np: np.ndarray) -> float:
        """Converts numpy inputs to torch tensors for the utility function."""
        # Move inputs to the correct device
        p_t = torch.from_numpy(p_np.astype(np.float32)).to(device)
        lam_t = torch.tensor(lam_float, dtype=torch.float32, device=device)
        x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
        
        # Call the PyTorch-based utility function
        utility_tensor = single_utility_from_env(p_t, lam_t, x_t) #
        
        # Return a standard Python float
        return utility_tensor.item()
    return numpy_utility_wrapper

# --- Evaluator 1: RL2O-CVaR Model (Dynamic Policy) ---

def evaluate_rl2o_cvar_model(
    model: nn.Module,
    val_dataset_q: np.ndarray,
    env_pool_list: List[Tuple[np.ndarray, float]],
    project_fn: Callable,
    S: float,
    gamma: float,
    device: torch.device,
    rng: Generator,
    n_env_eval: int,
    alpha: float
) -> Dict[str, float]:
    """
    Evaluates the trained RL2O-CVaR model.
    This is a "dynamic" policy, as the policy x changes for each test sample q.
    """
    metrics_model = evaluate_model(
        model=model,
        val_dataset_q=val_dataset_q,
        env_pool_list=env_pool_list,
        project_fn=project_fn,
        compute_utility_fn=single_utility_from_env, #
        S=S,
        gamma=gamma,
        device=device,
        rng=rng,
        n_env_eval=n_env_eval
    ) #
    
    # Re-format metrics for consistent reporting
    # Note: evaluate_model's 'VaR_{gamma}' is CVaR-loss at '1-gamma'
    metrics_model_agg = {
        'mean_utility': metrics_model['mean_utility'],
        'mean_loss': metrics_model['mean_loss_mean'],
        f'VaR_loss_{alpha:.2f}': metrics_model[f'VaR_loss_{gamma:.2f}'],
        f'CVaR_loss_{alpha:.2f}': metrics_model[f'CVaR_loss_{gamma:.2f}']
    }
    return metrics_model_agg

# --- Evaluator 2: Top-S Baseline (Static Policy) ---

def evaluate_top_s_baseline(
    p_bar_global: np.ndarray,
    S: float,
    env_pool_list: Sequence[Tuple[np.ndarray, float]],
    compute_utility_fn: Callable,
    n_env_eval: int,
    rng: Generator,
    alpha: float
) -> Dict[str, float]:
    """
    Evaluates the "Popularity Top-S" baseline.
    This is a "static" policy, as the policy x is fixed for all test samples.
    """
    S_int = int(round(S))  # Top-S policy needs an integer capacity
    
    # 2a. Create the single, fixed Top-S policy
    x_topS = popularity_deterministic_topS(p_bar_global, S_int) #
    
    # 2b. Evaluate the fixed policy x_topS against many environment samples
    u_samples_topS = evaluate_policy_on_env_pool(
        x=x_topS,
        env_pool=env_pool_list,
        compute_utility_fn=compute_utility_fn,
        n_sample=n_env_eval,
        rng=rng
    ) #
    
    # 2c. Calculate metrics from the resulting utility samples
    loss_samples_topS = 1.0 - u_samples_topS
    
    # Use the helper from model_eval.py to get VaR/CVaR for the *loss*
    metrics_topS = loss_tail_metrics(loss_samples_topS, alpha=alpha) #
    
    # Add the mean metrics
    metrics_topS['mean_utility'] = u_samples_topS.mean()
    metrics_topS['mean_loss'] = loss_samples_topS.mean()
    
    return metrics_topS

# --- Evaluator 3: New Method (Placeholder) ---

def evaluate_new_method_placeholder(
    p_bar_global: np.ndarray,
    test_dataset_q: np.ndarray,
    S: float,
    env_pool_list: Sequence[Tuple[np.ndarray, float]],
    compute_utility_fn: Callable,
    n_env_eval: int,
    rng: Generator,
    alpha: float
) -> Dict[str, float]:
    """
    This is a placeholder for your third method.
    Fill in the logic here to evaluate your new policy.
    """
    
    # TODO: Implement your new method here.
    # You need to generate a policy vector 'x_new' (shape M,).
    
    # --- OPTION A: If your new method is STATIC (like Top-S) ---
    # 1. Create your fixed policy vector 'x_new'
    #    (e.g., from p_bar_global or other statistics)
    #
    #    M = p_bar_global.shape[0]
    #    x_new = np.full((M,), S / M) # Example: a simple uniform policy
    #
    # 2. Evaluate it using 'evaluate_policy_on_env_pool'
    #    u_samples_new = evaluate_policy_on_env_pool(
    #        x=x_new,
    #        env_pool=env_pool_list,
    #        compute_utility_fn=compute_utility_fn,
    #        n_sample=n_env_eval,
    #        rng=rng
    #    )
    #
    # 3. Compute metrics
    #    loss_samples_new = 1.0 - u_samples_new
    #    metrics_new = loss_tail_metrics(loss_samples_new, alpha=alpha)
    #    metrics_new['mean_utility'] = u_samples_new.mean()
    #    metrics_new['mean_loss'] = loss_samples_new.mean()
    #    return metrics_new

    # --- OPTION B: If your new method is DYNAMIC (like the NN) ---
    # 1. You must loop over each 'q' in 'test_dataset_q'
    #    policy_results = []
    #    for q_np in test_dataset_q:
    #        # 2. Generate the policy 'x_new' for this 'q'
    #        x_new = my_new_policy_function(q_np, S, ...) 
    #
    #        # 3. Evaluate this 'x_new' against the env pool
    #        u_samples_for_q = evaluate_policy_on_env_pool(
    #            x=x_new,
    #            env_pool=env_pool_list,
    #            compute_utility_fn=compute_utility_fn,
    #            n_sample=n_env_eval, # Note: n_env_eval_model is usually smaller
    #            rng=rng
    #        )
    #        loss_samples_for_q = 1.0 - u_samples_for_q
    #
    #        # 4. Get metrics for this 'q'
    #        metrics_for_q = loss_tail_metrics(loss_samples_for_q, alpha=alpha)
    #        metrics_for_q['mean_utility'] = u_samples_for_q.mean()
    #        metrics_for_q['mean_loss'] = loss_samples_for_q.mean()
    #        policy_results.append(metrics_for_q)
    #
    # 5. Average the metrics across all 'q' samples
    #    df_results = pd.DataFrame(policy_results)
    #    metrics_new = df_results.mean().to_dict()
    #    return metrics_new
    
    # --- For now, returning NaNs ---
    print("     ... New method is a placeholder, returning NaN.")
    return {
        'mean_utility': np.nan,
        'mean_loss': np.nan,
        f'VaR_loss_{alpha:.2f}': np.nan,
        f'CVaR_loss_{alpha:.2f}': np.nan
    }


# --- Smoke Test / Example Usage ---
if __name__ == "__main__":
    """
    Run a small smoke test to demonstrate the new structure.
    This requires creating mock objects.
    """
    print("--- Running Smoke Test for final_comparison.py ---")
    
    # 1. Basic Setup
    M_FILES = 20
    S_CAPACITY = 5.0
    CVAR_GAMMA = 0.05
    N_TEST_Q = 10
    N_POOL = 1000
    
    rng = set_seed(42)
    device = get_device(prefer_cuda=False) # Use CPU for smoke test

    # 2. Create Mock Model
    class MockModel(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
        def forward(self, q):
            return self.linear(q)

    model = MockModel(in_dim=M_FILES + 1, out_dim=M_FILES).to(device)
    model.eval()

    # 3. Create Mock Data
    # Create a mock env_pool_list
    env_pool_list = []
    for _ in range(N_POOL):
        p = rng.dirichlet(np.ones(M_FILES))
        lam = rng.uniform(0.5, 3.0)
        env_pool_list.append((p.astype(np.float32), float(lam)))

    # Create a mock test dataset
    test_dataset_q = rng.random((N_TEST_Q, M_FILES + 1)).astype(np.float32)
    
    # 4. Run Comparison
    try:
        comparison_results_df = run_final_comparison(
            model=model,
            test_dataset_q=test_dataset_q,
            env_pool_list=env_pool_list,
            project_fn=project_capped_simplex, #
            S=S_CAPACITY,
            gamma=CVAR_GAMMA,
            device=device,
            rng=rng,
            n_env_eval_model=50,  # Small numbers for smoke test
            n_env_eval_baseline=200
        )

        # 5. Print Results
        print("\n--- Smoke Test Results ---")
        print(comparison_results_df)
        print("----------------------------")

    except Exception as e:
        print(f"Smoke teast failed: {e}")
        import traceback
        traceback.print_exc()