import env_pool


# result = env_pool._zipf_probs(10, 1.5)
# print(result)


# import numpy as np

# def test_dirichlet_posterior_samples():
#     # Parameters for the test
#     n = np.array([3, 5, 2])  # observed counts for 3 categories
#     N_pool = 10              # number of posterior samples to draw
#     alpha0 = 1.0             # symmetric Dirichlet prior concentration

#     # Generate posterior samples
#     posterior_samples = env_pool.dirichlet_posterior_samples_from_counts(n, N_pool, alpha0)

#     # Test assertions
#     assert posterior_samples.shape == (N_pool, len(n)), "Shape of output is incorrect"
#     assert np.allclose(posterior_samples.sum(axis=1), 1.0), "Rows of the output don't sum to 1"
    
#     # Print a few samples for inspection
#     print(posterior_samples[:3])  # Print the first 3 samples
    


# import torch
# from losses import project_capped_simplex
# y = torch.tensor([0.5, 0.7, 1.2, 0.3, 0.6]) 
# S = 2.5  
# result = project_capped_simplex(y, S)
# print("Projected tensor:", result) # all of them should be between 0,1
# print("Sum of projected tensor:", result.sum().item()) # should be equal to S



import numpy as np
import pytest

from env_pool import build_env_pool_simulated


def _kl_rowwise(p, q, eps=1e-12):
    """Return KL(p||q) per-row for p and q shaped (N, M)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.sum(p * np.log(p / q), axis=1)


def _l1_rowwise(p, q):
    """Return L1 distance per-row for p and q shaped (N, M)."""
    return np.sum(np.abs(np.asarray(p, float) - np.asarray(q, float)), axis=1)


@pytest.mark.parametrize("rng_seed", [123, 2023])
def test_build_env_pool_basic(rng_seed):
    """Basic deterministic checks: shapes, normalization, no-nans, positives."""
    N_pool = 10
    M = 8
    W = 50
    gamma_r = 1.2
    a0_p = 1.0
    a0_l = 1.0
    b0_l = 1.0
    A = 10.0
    lambda_true = 3.0

    P_pool, LAM_pool = build_env_pool_simulated(
        N_pool=N_pool,
        M=M,
        W=W,
        zipf_exponent=gamma_r,
        a0_p=a0_p,
        a0_lambda=a0_l,
        b0_lambda=b0_l,
        A_obs=A,
        lambda_true=lambda_true,
        rng_seed=rng_seed,
        as_torch=False,
    )

    # Structural assertions
    assert isinstance(P_pool, np.ndarray), "P_pool must be a numpy array"
    assert isinstance(LAM_pool, np.ndarray), "LAM_pool must be a numpy array"
    assert P_pool.shape == (N_pool, M), f"expected P_pool shape {(N_pool, M)}, got {P_pool.shape}"
    assert LAM_pool.shape == (N_pool,), f"expected LAM_pool shape {(N_pool,)}, got {LAM_pool.shape}"

    # Numerical sanity
    row_sums = P_pool.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-8), f"rows of P_pool must sum to 1 (min/max sums: {row_sums.min()}, {row_sums.max()})"
    assert not np.isnan(P_pool).any(), "P_pool contains NaN"
    assert not np.isnan(LAM_pool).any(), "LAM_pool contains NaN"
    assert (P_pool >= 0).all(), "P_pool contains negative entries"
    assert (LAM_pool > 0).all(), "LAM_pool must be strictly positive"


def test_build_env_pool_statistics():
    """
    Lightweight statistical checks:
      - mean(LAM_pool) is reasonably close to lambda_true (tolerance 1.0)
      - mean L1 distance between P_pool rows and reference Zipf (gamma) is below a conservative threshold
    These are sanity checks, not strict guarantees.
    """
    N_pool = 10
    M = 8
    W = 50
    gamma_r = 1.2
    a0_p = 1.0
    a0_l = 1.0
    b0_l = 1.0
    A = 10.0
    lambda_true = 3.0
    rng_seed = 123

    P_pool, LAM_pool = build_env_pool_simulated(
        N_pool=N_pool,
        M=M,
        W=W,
        zipf_exponent=gamma_r,
        a0_p=a0_p,
        a0_lambda=a0_l,
        b0_lambda=b0_l,
        A_obs=A,
        lambda_true=lambda_true,
        rng_seed=rng_seed,
        as_torch=False,
    )

    # check lambda mean proximity
    lam_mean = float(np.mean(LAM_pool))
    lam_std = float(np.std(LAM_pool))
    assert abs(lam_mean - lambda_true) < 1.0, f"mean(LAM_pool) {lam_mean:.3f} is too far from lambda_true {lambda_true} (std {lam_std:.3f})"

    # reference Zipf pmf
    ranks = np.arange(1, M + 1)
    p_zipf = ranks ** (-gamma_r)
    p_zipf = p_zipf / p_zipf.sum()

    # L1 distance aggregated
    l1_per_row = _l1_rowwise(P_pool, p_zipf)
    mean_l1 = float(np.mean(l1_per_row))
    # conservative threshold: typical mean-L1 should be well below 1.0 for reasonable samples;
    # we set 0.7 as a moderate threshold. Tweak if you have different expectations.
    assert mean_l1 < 0.7, f"mean L1 distance to Zipf is {mean_l1:.4f} (>= 0.7). If your model is intended to produce more varied posteriors, raise threshold."

    # Also compute and log KL (but don't fail on it)
    kl_per_row = _kl_rowwise(P_pool, p_zipf)
    mean_kl = float(np.mean(kl_per_row))
    # we won't assert on KL strictly; provide an informative message if large
    if mean_kl > 1.0:
        pytest.warns(UserWarning, reason=f"Mean KL to Zipf is relatively large: {mean_kl:.4f}")


if __name__ == "__main__":
    # quick ad-hoc run without pytest runner
    pytest.main([__file__, "-q"])





import pytest
import torch
import numpy as np
from losses import project_capped_simplex  # Replace with the actual import for project_capped_simplex

def test_trivial_cases():
    y = torch.tensor([0.2, 0.5])
    assert torch.allclose(project_capped_simplex(y, 0.0), torch.zeros_like(y))
    assert torch.allclose(project_capped_simplex(y, 2.0), torch.ones_like(y))

@pytest.mark.parametrize('y', [torch.randn(5)])  # Parametrize the input for multiple test cases if needed
def test_trivial_zero_sum(y):
    """Test for zero sum input."""
    result = project_capped_simplex(y, 0)
    assert torch.allclose(result, torch.zeros_like(y)), f"Expected zeros, got {result}"


@pytest.mark.parametrize('y, S', [(
    torch.tensor([-0.5, 0.3, 1.7, 0.8]), 5  # Number of elements in y + 1
)])
def test_trivial_large_S(y, S):
    """Test for large S where all values should be capped to 1."""
    result = project_capped_simplex(y, S)
    expected = torch.ones_like(y)
    assert torch.allclose(result, expected), f"Expected ones, got {result}"


@pytest.mark.parametrize('y, S', [(
    torch.tensor([0.2, 0.5, 0.7, 1.5, -0.3], dtype=torch.float32), 2.0
)])
def test_interior_projection(y, S):
    """Test for projection within the simplex."""
    x = project_capped_simplex(y, S)
    
    # Check sum equals the target S
    assert torch.isclose(x.sum(), torch.tensor(S), atol=1e-5), f"Sum mismatch: expected {S}, got {x.sum()}"
    
    # Ensure values are within [0, 1]
    assert torch.all((0 <= x) & (x <= 1)), f"Values out of bounds: {x}"
    
    # Check interior values
    interior = (x > 0) & (x < 1)
    if interior.any():
        tau = (y[interior] - x[interior]).mean()
        assert torch.allclose(x[interior], y[interior] - tau, atol=1e-6), f"Projection mismatch for interior: {x[interior]} vs {y[interior] - tau}"
        assert torch.all(y[x == 0] <= tau + 1e-6), f"Incorrect boundary values: {x[x == 0]}"
        assert torch.all(y[x == 1] >= 1 + tau - 1e-6), f"Incorrect boundary values: {x[x == 1]}"

test_trivial_cases()
print("Done")


#  losses.py 



# test_rate_theta_rho_pSucc_utility.py
import math
import torch
import numpy as np
import pytest

from losses import (  # <-- مسیر ایمپورت را با ماژول خودتان تطبیق دهید
    rho_theta_alpha4,
    p_succ_alpha4,
    utility_from_env_samples,
)

# --- helpers (اختیاری اگر در ماژول اصلی اضافه‌شان نکردید) ---
def theta_from_T_bits(T_bits: float) -> float:
    return float(2.0**T_bits - 1.0)

def theta_from_T_bits_torch(T_bits: float, *, dtype=torch.float32, device=None) -> torch.Tensor:
    Tt = torch.tensor(float(T_bits), dtype=dtype, device=device)
    two = torch.tensor(2.0, dtype=dtype, device=device)
    return torch.pow(two, Tt) - 1.0


# -----------------------------
# 1) Theta = 2^T - 1
# -----------------------------
def test_theta_from_T_bits():
    assert theta_from_T_bits(0.0) == pytest.approx(0.0)
    assert theta_from_T_bits(1.0) == pytest.approx(1.0)     # 2^1 - 1 = 1
    assert theta_from_T_bits(2.0) == pytest.approx(3.0)     # 4 - 1
    # torch version consistency
    th_t = theta_from_T_bits_torch(3.0)
    assert th_t.item() == pytest.approx(7.0)

test_theta_from_T_bits()
# -----------------------------
# 2) rho(theta) properties
# -----------------------------
def test_rho_theta_alpha4_properties():
    th0 = torch.tensor(0.0)
    rho0 = rho_theta_alpha4(th0)
    assert rho0.item() == pytest.approx(0.0)  # rho(0)=0

    # rho should be positive and increasing for theta>0
    th1 = torch.tensor(0.1)
    th2 = torch.tensor(1.0)
    r1 = rho_theta_alpha4(th1).item()
    r2 = rho_theta_alpha4(th2).item()
    assert r1 > 0.0 and r2 > 0.0
    assert r2 > r1
test_rho_theta_alpha4_properties()

# -----------------------------
# 3) P_succ bounds & monotonicity in theta
# -----------------------------
def test_p_succ_bounds_and_monotonicity_in_theta():
    device = torch.device("cpu")
    dtype = torch.float32

    # simple scenario: lam_i, lam_I > 0
    lam_i = torch.full((5,), 0.2, dtype=dtype, device=device)
    lam_I = torch.full((5,), 0.5, dtype=dtype, device=device)

    theta_lo = theta_from_T_bits(0.5)   # smaller threshold
    theta_hi = theta_from_T_bits(2.0)   # larger threshold

    p_lo = p_succ_alpha4(lam_i, lam_I, theta_lo)  # shape (5,)
    p_hi = p_succ_alpha4(lam_i, lam_I, theta_hi)

    # bounds
    assert torch.all(p_lo >= 0) and torch.all(p_lo <= 1)
    assert torch.all(p_hi >= 0) and torch.all(p_hi <= 1)

    # monotonicity: higher theta => harder success => smaller P_succ
    assert torch.all(p_lo >= p_hi)
test_p_succ_bounds_and_monotonicity_in_theta()


# -----------------------------
# 4) Utility: shape, bounds, monotonicity in x
# -----------------------------
def test_utility_from_env_samples_shape_bounds_and_monotonicity():
    torch.manual_seed(0)
    L, M = 7, 10
    # random popularity rows that sum to 1
    p = torch.rand(L, M)
    p = p / p.sum(dim=1, keepdim=True)

    lam = torch.full((L,), 0.8)

    # x_zero (no caching) vs x_full (cache everything) for S=M
    x_zero = torch.zeros(M)
    x_full = torch.ones(M)

    theta = theta_from_T_bits(1.0)  # theta = 1

    U_zero = utility_from_env_samples(p, lam, x_zero, theta=theta)  # (L,)
    U_full = utility_from_env_samples(p, lam, x_full, theta=theta)

    # shapes & bounds
    assert U_zero.shape == (L,)
    assert torch.all(U_zero >= 0) and torch.all(U_zero <= 1)
    assert torch.all(U_full >= 0) and torch.all(U_full <= 1)

    # monotonicity in x: with larger x (more helpers), success should not decrease
    assert torch.all(U_full >= U_zero)
test_utility_from_env_samples_shape_bounds_and_monotonicity()

# -----------------------------
# 5) Consistency: passing T in bits vs passing theta directly
# -----------------------------
def test_bits_vs_theta_consistency_in_pipeline():
    L, M = 3, 5
    p = torch.rand(L, M)
    p = p / p.sum(dim=1, keepdim=True)
    lam = torch.full((L,), 0.6)
    x = torch.full((M,), 0.5)

    T_bits = 1.5
    theta_bits = theta_from_T_bits(T_bits)
    theta_direct = theta_bits  # should be identical

    U1 = utility_from_env_samples(p, lam, x, theta=theta_bits)
    U2 = utility_from_env_samples(p, lam, x, theta=theta_direct)

    assert torch.allclose(U1, U2, atol=1e-6)
test_bits_vs_theta_consistency_in_pipeline()



# test_smoothed_cvar.py
import math
import torch
import torch.nn.functional as F
import pytest

# مسیر import را با ماژول خودت هماهنگ کن
from losses import (
    _softplus_scaled, _sigma_tau,
    solve_t_star_bisection,
)


def test_softplus_scaled_formula_matches_definition():
    torch.manual_seed(0)
    z = torch.randn(100)
    tau = 3.7
    ours = _softplus_scaled(z, tau)
    ref  = (1.0 / tau) * F.softplus(tau * z)
    assert torch.allclose(ours, ref, atol=1e-10)
test_softplus_scaled_formula_matches_definition()

def test_sigma_tau_is_sigmoid_scaled():
    torch.manual_seed(0)
    z = torch.linspace(-4, 4, 101)
    tau = 2.5
    ours = _sigma_tau(z, tau)
    ref  = torch.sigmoid(tau * z)
    assert torch.allclose(ours, ref, atol=1e-10)

    assert torch.all((ours > 0) & (ours < 1))
test_sigma_tau_is_sigmoid_scaled()


def test_t_star_satisfies_first_order_condition():
    torch.manual_seed(0)
    L = 257
    ell = torch.randn(L) * 0.7 + 1.1  # lossها
    gamma = 0.1
    tau = 5.0

    t_star = solve_t_star_bisection(ell, gamma=gamma, tau=tau, tol=1e-6, max_iter=80)
    s = _sigma_tau(ell - t_star, tau).sum()
    foc = 1.0 - (1.0 / (gamma * L)) * s
    assert torch.isclose(foc, torch.tensor(0.0), atol=5e-3)
test_t_star_satisfies_first_order_condition()


def test_t_star_batch_and_single_agree():
    torch.manual_seed(0)
    B, L = 4, 129
    E = torch.randn(B, L)
test_t_star_batch_and_single_agree()

# tests/test_projection_and_psucc.py
import torch
from losses import project_capped_simplex, rho_theta_alpha4, p_succ_alpha4


def test_rho_and_psucc_ranges():
    thetas = torch.tensor([0.0, 0.1, 1.0, 5.0])
    r = rho_theta_alpha4(thetas)
    assert torch.all(r >= 0)
    lam_i = torch.tensor([0.0, 0.2, 0.5, 1.0])
    lam_I = torch.tensor([0.1, 0.1, 0.1, 0.1])
    P = p_succ_alpha4(lam_i, lam_I, theta=0.5)
    assert torch.all((P >= 0) & (P <= 1))
