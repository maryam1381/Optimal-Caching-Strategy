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



import torch
from losses import project_capped_simplex

def test_trivial_cases():
    y = torch.tensor([0.2, 0.5])
    assert torch.allclose(project_capped_simplex(y, 0.0), torch.zeros_like(y))
    assert torch.allclose(project_capped_simplex(y, 2.0), torch.min(y, torch.ones_like(y)))

def test_projection_sum_and_bounds():
    y = torch.tensor([0.2, 1.5, -0.3, 0.8])
    S = 1.0
    x = project_capped_simplex(y, S)
    assert torch.all(x >= 0) and torch.all(x <= 1)
    assert torch.isclose(x.sum(), torch.tensor(S))

test_trivial_cases()
test_projection_sum_and_bounds()
print("Done")