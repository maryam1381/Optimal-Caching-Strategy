# src/env_pool.py
"""
Environment-posterior sampling utilities for digital-twin experiments.

Provides:
 - Dirichlet posterior sampling for file popularity (via Gamma draws + normalization).
 - Gamma posterior sampling for PPP intensity lambda (conjugate to Poisson counts).
 - Builders to create an environment pool (P_pool, LAM_pool) for training/evaluation.

Author: Assistant (adapt for your project)
"""

from typing import Optional, Tuple
import numpy as np
import torch
import os


def _zipf_probs(M: int, gamma_r: float) -> np.ndarray:
    """Return a normalized Zipf probability vector for ranks 1..M."""
    ranks = np.arange(1, M + 1, dtype=float)
    probs = 1.0 / (ranks ** float(gamma_r))
    probs /= probs.sum()
    return probs


def dirichlet_posterior_samples_from_counts(
    n: np.ndarray,
    N_pool: int,
    alpha0: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Draw i.i.d. samples from Dirichlet(alpha0 + n) using Gamma draws + normalization.

    Args:
        n: array shape (M,) observed counts (non-negative integers).
        N_pool: number of posterior samples to draw.
        alpha0: symmetric Dirichlet prior concentration.
        rng: optional numpy.random.Generator for reproducibility.

    Returns:
        p_samples: np.ndarray shape (N_pool, M) with rows summing to 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = np.asarray(n, dtype=float)
    M = n.size
    # Gamma shape parameters (alpha0 + n_i) for each category
    shapes = (alpha0 + n).astype(float)  # shape (M,)
    # Draw Gamma variates: shape (N_pool, M)
    # Use broadcasting: draw N_pool x M independent Gammas
    g = rng.gamma(shape=shapes[None, :], scale=1.0, size=(N_pool, M))
    # Normalize rows to sum to 1
    row_sums = g.sum(axis=1, keepdims=True)
    p_samples = g / row_sums
    return p_samples


def lambda_posterior_samples_from_count(
    K: int,
    N_pool: int,
    a0: float = 1.0,
    b0: float = 1.0,
    A: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Draw i.i.d. samples from Gamma(shape=a0 + K, rate=b0 + A) posterior.
    Note: numpy.random.gamma uses shape & scale (scale = 1/rate).

    Args:
        K: observed count (integer).
        N_pool: number of posterior samples.
        a0,b0: prior hyperparameters (shape, rate).
        A: observed area (so Poisson mean = lambda * A).
        rng: optional numpy.random.Generator.

    Returns:
        lambda_samples: np.ndarray shape (N_pool,) with positive floats.
    """
    if rng is None:
        rng = np.random.default_rng()

    shape = float(a0 + K)
    rate = float(b0 + A)   # rate parameter
    scale = 1.0 / rate     # numpy uses scale
    lambdas = rng.gamma(shape=shape, scale=scale, size=(N_pool,))
    return lambdas


def build_env_pool_from_obs(
    n: np.ndarray,
    K: int,
    N_pool: int,
    alpha0_p: float = 1.0,
    a0_lambda: float = 1.0,
    b0_lambda: float = 1.0,
    A_obs: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    as_torch: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build environment pool (popularity vectors and lambda samples) conditioned on measurements.

    Args:
        n: observed request counts array (M,)
        K: observed number of users in the monitoring area
        N_pool: number of i.i.d. posterior samples to create
        alpha0_p: Dirichlet prior concentration (popularity)
        a0_lambda, b0_lambda: Gamma prior hyperparams for lambda (shape, rate)
        A_obs: monitoring area (used in posterior for lambda)
        rng: optional numpy.random.Generator for reproducibility
        as_torch: if True, return torch tensors (float32), else numpy arrays
        device: torch.device to map tensors to (if as_torch=True). Default CPU.

    Returns:
        (P_pool, LAM_pool)
          - P_pool: shape (N_pool, M) row-stochastic popularities
          - LAM_pool: shape (N_pool,) intensities (positive floats)
        Types depend on as_torch flag.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ensure n is integer array
    n = np.asarray(n, dtype=int)
    assert n.ndim == 1, "n must be 1-D counts array of length M"
    M = n.size

    # Popularity posterior samples
    P_pool = dirichlet_posterior_samples_from_counts(n=n, N_pool=N_pool, alpha0=alpha0_p, rng=rng)

    # Lambda posterior samples (posterior shape a0 + K, posterior rate b0 + A)
    LAM_pool = lambda_posterior_samples_from_count(K=K, N_pool=N_pool,
                                                  a0=a0_lambda, b0=b0_lambda, A=A_obs, rng=rng)

    if as_torch:
        device = device if device is not None else torch.device("cpu")
        P_pool_t = torch.from_numpy(P_pool.astype(np.float32)).to(device)
        LAM_pool_t = torch.from_numpy(LAM_pool.astype(np.float32)).to(device)
        return P_pool_t, LAM_pool_t
    else:
        return P_pool, LAM_pool


def build_env_pool_simulated(
    N_pool: int,
    M: int,
    W: int,
    zipf_exponent: float,
    a0_p: float = 1.0,
    a0_lambda: float = 1.0,
    b0_lambda: float = 1.0,
    A_obs: float = 1.0,
    lambda_true: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    as_torch: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single measurement (counts n and count K) from a Zipf+Poisson truth,
    then form posterior samples conditional on that measurement.

    Args:
        N_pool, M, W, zipf_exponent: simulation parameters
        priors and obs area : a0_p, a0_lambda, b0_lambda, A_obs
        lambda_true: ground-truth user density (used to simulate observed K)
        rng: optional numpy.random.Generator for reproducibility
        as_torch, device: output format

    Returns:
        (P_pool, LAM_pool) same as build_env_pool_from_obs
    """
    if rng is None:
        rng = np.random.default_rng()

    # true popularity (Zipf)
    p_true = _zipf_probs(M, zipf_exponent)

    # observed request counts (Multinomial(W, p_true))
    if W > 0:
        n = rng.multinomial(W, p_true)
    else:
        n = np.zeros(M, dtype=int)

    # observed number of users in area A_obs (Poisson)
    K = rng.poisson(lam=lambda_true * A_obs)

    # build posterior samples conditioned on the simulated observations
    return build_env_pool_from_obs(n=n, K=int(K), N_pool=N_pool,
                                   alpha0_p=a0_p, a0_lambda=a0_lambda,
                                   b0_lambda=b0_lambda, A_obs=A_obs,
                                   rng=rng, as_torch=as_torch, device=device)


# small helpers --------------------------------------------------------------

def sample_env_batch(
    P_pool: torch.Tensor,
    LAM_pool: torch.Tensor,
    inds: np.ndarray,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pick subset of environment pool by indices and return tensors on the given device.

    Args:
        P_pool: torch tensor (N_pool, M)
        LAM_pool: torch tensor (N_pool,)
        inds: 1-D numpy integer array or torch tensor of indices length L
        device: desired torch.device (if None keep current)

    Returns:
        (P_batch, LAM_batch) with shapes (L, M) and (L,)
    """
    if isinstance(inds, np.ndarray):
        inds_t = torch.from_numpy(inds).long()
    elif isinstance(inds, torch.Tensor):
        inds_t = inds.long()
    else:
        inds_t = torch.tensor(list(inds), dtype=torch.long)

    device = device if device is not None else P_pool.device
    inds_t = inds_t.to(device)
    P_batch = P_pool[inds_t]
    LAM_batch = LAM_pool[inds_t]
    return P_batch, LAM_batch


def save_env_pool(path: str, P_pool: np.ndarray, LAM_pool: np.ndarray) -> None:
    """
    Save environment pool to .npz. Accept numpy arrays. Overwrites existing file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, P_pool=P_pool, LAM_pool=LAM_pool)


def load_env_pool(path: str, as_torch: bool = True, device: Optional[torch.device] = None):
    """
    Load environment pool saved by save_env_pool.
    Returns torch tensors (if as_torch=True) or numpy arrays.
    """
    with np.load(path) as data:
        P = data["P_pool"]
        LAM = data["LAM_pool"]
    if as_torch:
        device = device if device is not None else torch.device("cpu")
        return torch.from_numpy(P.astype(np.float32)).to(device), torch.from_numpy(LAM.astype(np.float32)).to(device)
    else:
        return P, LAM


# quick unit-test / smoke-test
def _test_build_env_pool():
    N_pool = 10
    M = 8
    W = 50
    gamma_r = 1.2
    a0_p = 1.0
    a0_l = 1.0
    b0_l = 1.0
    A = 10.0
    lambda_true = 3.0
    rng = np.random.default_rng(123)
    P_pool, LAM_pool = build_env_pool_simulated(
        N_pool=N_pool, M=M, W=W, zipf_exponent=gamma_r,
        a0_p=a0_p, a0_lambda=a0_l, b0_lambda=b0_l, A_obs=A,
        lambda_true=lambda_true, rng=rng, as_torch=False
    )
    # print("P pool : ",P_pool[:5])
    # print("Lam pool : ",LAM_pool[:5])
    assert P_pool.shape == (N_pool, M)
    assert LAM_pool.shape == (N_pool,)
    # row-stochastic check
    s = P_pool.sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-6)
    assert np.all(LAM_pool > 0)
    print("env_pool simulated test passed.")

if __name__ == "__main__":
    _test_build_env_pool()

