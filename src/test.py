# import math
# import torch
# import pytest

# from losses import _sigma_tau, solve_t_star_bisection, smoothed_cvar_loss_from_utilities


# def empirical_upper_tail_quantile(ell: torch.Tensor, gamma: float) -> float:
#     """
#     Empirical (1 - gamma) quantile for losses.
#     """
#     x = torch.sort(ell.flatten()).values.cpu().numpy()
#     N = x.size
#     q = 1.0 - gamma
#     k = max(0, min(N - 1, int(math.ceil(q * N) - 1)))
#     return float(x[k])


# def test_sigma_tau_limits_behave_like_indicator():
#     z = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
#     s_small = _sigma_tau(z, tau=0.5)
#     s_big = _sigma_tau(z, tau=1e6)
#     # At very large tau, behaves like 1_{z>0} with ~0.5 at 0
#     assert s_big[0] < 1e-6 and s_big[1] < 1e-3
#     assert abs(float(s_big[2]) - 0.5) < 1e-6
#     assert 1.0 - float(s_big[3]) < 1e-3 and 1.0 - float(s_big[4]) < 1e-6
#     # Monotonic in tau
#     assert s_small[3] < s_big[3] and s_small[1] > s_big[1]


# @pytest.mark.parametrize("gamma", [0.05, 0.1, 0.2, 0.4])
# def test_t_star_matches_empirical_quantile_in_hard_limit(gamma):
#     torch.manual_seed(0)
#     ell = torch.linspace(0.0, 1.0, steps=101)
#     tau = 1e6
#     t_star = solve_t_star_bisection(ell, gamma=gamma, tau=tau, tol=1e-10, max_iter=200)
#     t_emp = empirical_upper_tail_quantile(ell, gamma=gamma)
#     assert abs(float(t_star) - t_emp) < 1e-3


# def test_residual_near_zero_at_returned_t():
#     torch.manual_seed(123)
#     B, L = 3, 257
#     U = torch.rand(B, L)
#     ell = 1.0 - U
#     gamma, tau = 0.05, 5.0
#     t_star = solve_t_star_bisection(ell, gamma=gamma, tau=tau, tol=1e-6, max_iter=200)

#     def residual(tvec):
#         tcol = tvec.view(-1, 1)
#         sig = _sigma_tau(ell - tcol, tau=tau).sum(dim=1)
#         return 1.0 - (1.0 / (gamma * float(L))) * sig

#     g = residual(t_star)
#     assert torch.max(torch.abs(g)).item() < 5e-4


# def test_t_star_converges_to_quantile_as_tau_increases():
#     torch.manual_seed(1234)
#     ell = torch.sort(torch.rand(1001))[0]
#     gamma = 0.1
#     taus = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0]
#     t_emp = empirical_upper_tail_quantile(ell, gamma=gamma)
#     errors = []
#     prev_err = None
#     for tau in taus:
#         t_star = solve_t_star_bisection(ell, gamma=gamma, tau=tau, tol=1e-6, max_iter=200)
#         err = abs(float(t_star) - t_emp)
#         errors.append(err)
#         if prev_err is not None:
#             # error should shrink overall
#             assert err <= max(prev_err, errors[0]) + 5e-3
#         prev_err = err
#     assert errors[-1] < 2e-3


# def test_batch_and_single_agree_and_shapes():
#     torch.manual_seed(999)
#     ell_row = torch.rand(257)
#     ell_batch = torch.stack([ell_row.clone(), ell_row.clone(), ell_row.clone()], dim=0)
#     gamma, tau = 0.2, 3.0

#     t_single = solve_t_star_bisection(ell_row, gamma=gamma, tau=tau)
#     t_batch = solve_t_star_bisection(ell_batch, gamma=gamma, tau=tau)
#     assert t_batch.shape == (3,)
#     assert torch.allclose(t_batch, torch.full((3,), t_single, dtype=t_batch.dtype), atol=1e-6, rtol=0)


# def test_smoothed_cvar_loss_runs_and_consistent_with_solver():
#     torch.manual_seed(2025)
#     B, L = 4, 311
#     U = torch.rand(B, L)
#     gamma, tau = 0.05, 4.0
#     loss_mean, per = smoothed_cvar_loss_from_utilities(U, gamma=gamma, tau=tau, per_sample=True)
#     assert loss_mean.ndim == 0 and per.shape == (B,)
#     ell = 1.0 - U
#     t_star = solve_t_star_bisection(ell, gamma=gamma, tau=tau)
#     tcol = t_star.view(-1, 1)
#     g = 1.0 - (1.0 / (gamma * float(L))) * _sigma_tau(ell - tcol, tau=tau).sum(dim=1)
#     assert torch.max(torch.abs(g)).item() < 5e-4


#  pytest -q test.py



# import numpy as np
# import torch
# import pytest

# from eval import popularity_deterministic_topS, mean_opt_plug_in
# from losses import project_capped_simplex

# def test_topS_basic():
#     p = np.array([0.1, 0.9, 0.3, 0.5])
#     x = popularity_deterministic_topS(p, S=2)
#     # Should pick indices 1 and 3 (values 0.9, 0.5)
#     assert x.sum() == 2
#     assert x[1] == 1.0 and x[3] == 1.0

# def test_topS_zero_S():
#     p = np.array([0.2, 0.8, 0.5])
#     x = popularity_deterministic_topS(p, S=0)
#     assert np.allclose(x, 0)

# def test_topS_large_S():
#     p = np.array([0.2, 0.8, 0.5])
#     x = popularity_deterministic_topS(p, S=10)  # larger than len(p)
#     assert np.allclose(x, 1)  # all should be cached

# def dummy_utility(p, lam, x):
#     # p and lam are numpy arrays in our test setup, so convert to tensors
#     if not torch.is_tensor(p):
#         p = torch.tensor(p, dtype=torch.float32)
#     if not torch.is_tensor(lam):
#         lam = torch.tensor(lam, dtype=torch.float32)
#     # ensure they’re on the same device as x
#     p = p.to(x.device)
#     lam = lam.to(x.device)

#     # Differentiable linear utility
#     return (p * x).sum() - lam * x.sum()

# # def dummy_utility(p, lam, x):
# #     # p, lam, x are torch tensors
# #     return (p * x).sum() - lam * x.sum()


# def test_mean_opt_plug_in_linear():
#     p = np.array([0.1, 0.9, 0.3], dtype=np.float32)
#     lam = 0.0
#     S = 1
#     M = len(p)

#     x_opt = mean_opt_plug_in(
#         p_bar=p,
#         lambda_bar=lam,
#         S=S,
#         compute_utility_fn=dummy_utility,
#         project_fn=project_capped_simplex,
#         M=M,
#         steps=100,
#         lr=0.1,
#         device="cpu",
#         verbose=False
#     )

#     assert np.isclose(x_opt.sum(), 1.0, atol=1e-3)
#     assert np.argmax(x_opt) == 1  # should choose index 1 (highest p)

# def test_mean_opt_plug_in_zero_capacity():
#     p = np.array([0.5, 0.2, 0.3], dtype=np.float32)
#     lam = 0.0
#     S = 0
#     M = len(p)

#     x_opt = mean_opt_plug_in(
#         p_bar=p,
#         lambda_bar=lam,
#         S=S,
#         compute_utility_fn=dummy_utility,
#         project_fn=project_capped_simplex,
#         M=M,
#         steps=50,
#         lr=0.1,
#         device="cpu",
#     )

#     assert np.allclose(x_opt, 0, atol=1e-6)



# import numpy as np
# import torch
# from model_eval import evaluate_all_policies

# # Simple capped simplex projection for tests
# def toy_project_fn(x, S):
#     x = np.clip(x, 0, 1)
#     if x.sum() > S:
#         x = x / x.sum() * S
#     return x

# # Simple linear utility
# def toy_compute_utility(p, lam, x):
#     if not torch.is_tensor(p):
#         p = torch.tensor(p, dtype=torch.float32, device=x.device)
#     if not torch.is_tensor(lam):
#         lam = torch.tensor(lam, dtype=torch.float32, device=x.device)
#     return (p * x).sum() - lam * x.sum()

# def test_evaluate_all_policies_basic():
#     # toy dataset q
#     val_dataset_q = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

#     # toy environment pool: two envs with slightly different p
#     env_pool_list = [
#         (np.array([0.8, 0.2], dtype=np.float32), 0.0),
#         (np.array([0.1, 0.9], dtype=np.float32), 0.0),
#     ]

#     S = 1
#     gamma = 0.05
#     device = "cpu"

#     # Define policies
#     def always_first(_q):
#         return np.array([1.0, 0.0], dtype=np.float32)

#     def always_second(_q):
#         return np.array([0.0, 1.0], dtype=np.float32)

#     policies = {
#         "First": always_first,
#         "Second": always_second
#     }

#     results = evaluate_all_policies(
#         policies_to_evaluate=policies,
#         val_dataset_q=val_dataset_q,
#         env_pool_list=env_pool_list,
#         project_fn=toy_project_fn,
#         compute_utility_fn=toy_compute_utility,
#         S=S,
#         gamma=gamma,
#         device=device,
#         rng=np.random.default_rng(0),
#         n_env_eval=10
#     )

#     # Check that both policies are in results
#     assert "First" in results
#     assert "Second" in results

#     # Each should have standard keys
#     for metrics in results.values():
#         assert "mean_utility_mean" in metrics
#         assert "CVaR_0.05" in metrics
#         assert isinstance(metrics["mean_utility_mean"], float)

# def test_evaluate_all_policies_relative_performance():
#     # Environment pool where index 0 is always better
#     env_pool_list = [(np.array([1.0, 0.0], dtype=np.float32), 0.0)]
#     val_dataset_q = [np.array([1.0, 0.0])]

#     S = 1
#     gamma = 0.05
#     device = "cpu"

#     def always_first(_q):
#         return np.array([1.0, 0.0], dtype=np.float32)

#     def always_second(_q):
#         return np.array([0.0, 1.0], dtype=np.float32)

#     policies = {"First": always_first, "Second": always_second}

#     results = evaluate_all_policies(
#         policies_to_evaluate=policies,
#         val_dataset_q=val_dataset_q,
#         env_pool_list=env_pool_list,
#         project_fn=toy_project_fn,
#         compute_utility_fn=toy_compute_utility,
#         S=S,
#         gamma=gamma,
#         device=device,
#         rng=np.random.default_rng(0),
#         n_env_eval=5
#     )

#     # First should strictly outperform second
#     assert results["First"]["mean_utility_mean"] > results["Second"]["mean_utility_mean"]



# tests/test_eval_equivalence.py
import numpy as np
import torch

from model_eval import evaluate_all_policies , evaluate_model

# --- Helper: capped-simplex projection that enforces 0<=x<=1 and sum(x)<=S ---
def capped_simplex_project(x_np: np.ndarray, S: float) -> np.ndarray:
    """
    Project x onto { x in [0,1]^M, sum x <= S } using simple bisection.
    """
    x = np.clip(x_np, 0.0, 1.0)
    total = x.sum()
    if total <= S + 1e-8:
        return x
    # Find tau s.t. sum(clip(x - tau, 0, 1)) = S
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        y = np.clip(x - mid, 0.0, 1.0)
        if y.sum() > S:
            lo = mid
        else:
            hi = mid
    return np.clip(x - hi, 0.0, 1.0)

# --- Helper: linear utility (torch-friendly; returns a scalar tensor) ---
def linear_utility(p, lam, x):
    """
    Utility = (p * x).sum() - lam * x.sum()
    p, lam, x must be torch tensors; evaluator will pass torch tensors for p/lam/x.
    """
    # Ensure torch tensors are on the same device
    if not torch.is_tensor(p):
        p = torch.tensor(p, dtype=torch.float32, device=x.device)
    if not torch.is_tensor(lam):
        lam = torch.tensor(lam, dtype=torch.float32, device=x.device)
    return (p * x).sum() - lam * x.sum()

def _build_shared_data(seed: int = 123, M: int = 8, IN: int = 4):
    """
    Build identical val_dataset_q and env_pool_list to be reused by both evaluators.
    """
    rng = np.random.default_rng(seed)
    val_q = [rng.random(IN).astype(np.float32) for _ in range(30)]
    env_pool = [(rng.random(M).astype(np.float32), 0.0) for _ in range(300)]
    return val_q, env_pool

class ToyModel(torch.nn.Module):
    """
    Simple deterministic model to reduce stochastic variation:
    a single linear layer initialized to zeros.
    Output dimension must be M.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        return self.lin(x)

def _nearly_equal(a: float, b: float, tol: float = 1e-2) -> bool:
    return abs(float(a) - float(b)) < tol

def test_evaluate_single_vs_multi_equivalence():
    """
    Both evaluate_model (single policy) and evaluate_all_policies (dict with the same model)
    should return (nearly) identical aggregate metrics when:
      - same projection function is used
      - same RNG seed and n_env_eval are used
      - same val_dataset_q and env_pool_list are used
    """
    # Problem sizes and constants
    M, IN = 8, 4
    S = 3.0
    gamma = 0.05
    device = "cpu"
    n_env_eval = 200

    # Build shared data
    val_q, env_pool = _build_shared_data(seed=2024, M=M, IN=IN)

    # Deterministic model
    model = ToyModel(in_dim=IN, out_dim=M)

    # Use separate RNGs with the same seed so both evaluators sample identically
    rng_single = np.random.default_rng(42)
    rng_multi  = np.random.default_rng(42)

    # Run single-policy evaluator
    single = evaluate_model(
        model=model,
        val_dataset_q=val_q,
        env_pool_list=env_pool,
        project_fn=capped_simplex_project,
        compute_utility_fn=linear_utility,
        S=S,
        gamma=gamma,
        device=device,
        n_env_eval=n_env_eval,
        rng=rng_single
    )

    # Run multi-policy evaluator with the same model
    multi_all = evaluate_all_policies(
        policies_to_evaluate={"Model": model},
        val_dataset_q=val_q,
        env_pool_list=env_pool,
        project_fn=capped_simplex_project,
        compute_utility_fn=linear_utility,
        S=S,
        gamma=gamma,
        device=device,
        n_env_eval=n_env_eval,
        rng=rng_multi
    )
    multi = multi_all["Model"]

    # Compare overlapping metric keys with a reasonable tolerance
    common_keys = set(single.keys()) & set(multi.keys())
    assert len(common_keys) > 0, "No common metric keys to compare."

    # Required key: mean_utility_mean should match closely
    assert _nearly_equal(single["mean_utility_mean"], multi["mean_utility_mean"], tol=1e-2), \
        f"mean_utility_mean mismatch: single={single['mean_utility_mean']} vs multi={multi['mean_utility_mean']}"

    # Optional: compare a few other common metrics if present
    for k in ["mean_utility_std", "mean_loss_mean"]:
        if k in common_keys:
            assert _nearly_equal(single[k], multi[k], tol=2e-2), f"{k} mismatch."

    # If you compute loss right-tail metrics (preferred for risk-on-loss):
    # Use whatever labels exist in your codebase; here we check both options safely.
    if "VaR_loss_0.95" in common_keys and "CVaR_loss_0.95" in common_keys:
        assert _nearly_equal(single["VaR_loss_0.95"],  multi["VaR_loss_0.95"],  tol=2e-2)
        assert _nearly_equal(single["CVaR_loss_0.95"], multi["CVaR_loss_0.95"], tol=2e-2)
        # Monotonicity sanity for loss right tail:
        assert multi["CVaR_loss_0.95"] >= multi["VaR_loss_0.95"] - 1e-9
    elif "VaR_0.05" in common_keys and "CVaR_0.05" in common_keys:
        # If your code reports utility left-tail (less common in loss-focused work)
        assert _nearly_equal(single["VaR_0.05"],  multi["VaR_0.05"],  tol=2e-2)
        assert _nearly_equal(single["CVaR_0.05"], multi["CVaR_0.05"], tol=2e-2)
        # Monotonicity sanity for utility left tail:
        assert multi["CVaR_0.05"] <= multi["VaR_0.05"] + 1e-9
