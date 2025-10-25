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



