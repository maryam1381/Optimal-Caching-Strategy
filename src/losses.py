# losses.py
"""
Loss and utility helpers for CVaR training of robust D2D caching policies.

Requires: torch, numpy.

Main functions:
 - project_capped_simplex(y, S): project vector y onto {x in [0,1]^M : sum x <= S}
 - p_succ_alpha4(lam_i, lam_I, theta, P_t, N0, mu): closed-form per-file success prob (alpha=4)
 - utility_from_env_samples(p_samples, lam_samples, x, ...): compute U(x; env^{(s)}) for many envs
 - smoothed_cvar_loss_from_utilities(U_batch, gamma, tau, ...): compute mean CVaR loss over a batch
"""

from typing import Optional, Tuple
import math

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------
# Projection onto capped simplex
# ----------------------------
def project_capped_simplex(y: torch.Tensor, S: float, tol: float = 1e-6, max_iter: int = 60) -> torch.Tensor:
    """
    Euclidean projection of y onto { x in [0,1]^M : sum_i x_i <= S }.

    Implements a scalar bisection on theta for x_i = clip(y_i - theta, 0,1).
    This returns a tensor of same shape as y.

    Parameters
    ----------
    y : torch.Tensor (M,) or (batch, M)
        Input (unconstrained) vector(s).
    S : float
        Sum upper-bound (0 <= S <= M).
    tol : float
        Stopping tolerance for theta.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    x : torch.Tensor
        Projected vector(s), same dtype/device as y.
    """
    single = (y.dim() == 1)
    y_ = y.unsqueeze(0) if single else y  # (B, M)
    device, dtype = y_.device, y_.dtype
    B, M = y_.shape

    # trivial cases
    if S <= 0:
        return torch.zeros_like(y)
    if S >= float(M):
        return torch.ones_like(y)

    # clamp to [0,1]; rows already within the cap are done
    y_clamped = torch.clamp(y_, 0.0, 1.0)
    row_sums = y_clamped.sum(dim=1)
    mask_done = (row_sums <= S)
    x = y_clamped.clone()

    need_proj_idx = (~mask_done).nonzero(as_tuple=False).squeeze(1)
    if need_proj_idx.numel() == 0:
        return x.squeeze(0) if single else x

    S_t = torch.as_tensor(S, dtype=dtype, device=device)

    # process only rows that require projection
    for idx in need_proj_idx.tolist():
        vec = y_[idx]  # shape (M,)

        # torch scalars (0-D tensors) for bounds
        theta_lo = (vec - 1.0).min()    # ensures some slack
        theta_hi = vec.max()

        # fixed-iteration bisection fully in torch
        for _ in range(max_iter):
            theta = 0.5 * (theta_lo + theta_hi)
            candidate = torch.clamp(vec - theta, 0.0, 1.0)
            s = candidate.sum()  # 0-D tensor

            # update bounds without Python branching
            greater = (s > S_t)
            theta_lo = torch.where(greater, theta, theta_lo)
            theta_hi = torch.where(greater, theta_hi, theta)

        theta = 0.5 * (theta_lo + theta_hi)
        x[idx] = torch.clamp(vec - theta, 0.0, 1.0)

    return x.squeeze(0) if single else x


# ----------------------------
# Closed-form P_succ for alpha = 4 and Rayleigh interferers
# ----------------------------
def _Q_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian complementary CDF Q(x) = 0.5 * erfc(x/sqrt(2)) = 0.5*(1 - erf(x/sqrt(2))).

    Implemented via torch.erf which is widely available.
    """
    return 0.5 * (1.0 - torch.erf(x / math.sqrt(2.0)))

def rho_theta_alpha4(theta: float or torch.Tensor) -> torch.Tensor:
    """
    Geometry/fading factor rho(theta, 4) = sqrt(theta) * (pi/2 - atan(1/sqrt(theta))).
    Accepts scalar or tensor.
    """
    # Cast to tensor
    if not torch.is_tensor(theta):
        theta_t = torch.tensor(float(theta))
    else:
        theta_t = theta
    # Prevent non-positive numerical issues
    # for theta == 0, rho=0.
    safe = torch.clamp(theta_t, min=0.0)
    return torch.sqrt(safe) * (0.5 * math.pi - torch.atan(1.0 / torch.sqrt(safe + 1e-30)))


def _Q_torch_safe(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Numerically stable Gaussian complementary CDF Q(z).
    Uses asymptotic approximation for large z.
    """
    standard = 0.5 * (1.0 - torch.erf(z / math.sqrt(2.0)))
    asymptotic = torch.exp(-z**2 / 2) / (z * math.sqrt(2 * math.pi))
    use_asymptotic = z > 6.0
    q = torch.where(use_asymptotic, asymptotic, standard)
    return torch.clamp(q, min=eps)

def p_succ_alpha4(
    lam_i: torch.Tensor,
    lam_I: torch.Tensor,
    theta: float,
    P_t: float = 1.0,
    N0: float = 1e-9,
    mu: float = 1.0,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute per-file success probability under alpha=4 closed-form kernel.
    Uses log-domain computation for numerical stability.
    """
    device = lam_i.device
    dtype = lam_i.dtype

    # Compute scalar b
    b = mu * theta * (N0 / P_t)
    b_t = torch.tensor(max(b, eps), dtype=dtype, device=device)

    # Compute a = π * (lam_i + lam_I * rho)
    theta_t = torch.tensor(theta, dtype=dtype, device=device)
    rho = rho_theta_alpha4(theta_t)  # scalar
    a = math.pi * (lam_i + lam_I * rho)

    # Compute z = a / sqrt(2b)
    denom = torch.sqrt(2.0 * b_t)
    z = torch.clamp(a / denom, max=12.0)  # clamp to avoid log(0)

    # Compute log(P_succ)
    pref = (math.pi ** 1.5) * lam_i / torch.sqrt(b_t)
    exponent = torch.clamp((a ** 2) / (4.0 * b_t), max=50.0)
    log_P_succ = torch.log(pref + eps) + exponent + torch.log(_Q_torch_safe(z, eps))
    log_P_succ = torch.clamp(log_P_succ, max=0.0) 
    # Final success probability
    P_succ = torch.exp(log_P_succ)
    return torch.clamp(P_succ, min=0.0, max=1.0)

# ----------------------------
# Utility evaluation from a single x and many env samples
# ----------------------------
def utility_from_env_samples(
    p_samples: torch.Tensor,
    lam_samples: torch.Tensor,
    x: torch.Tensor,
    theta: float = 1,
    lambda_I_factor: float = 1.0,
    P_t: float = 1.0,
    N0: float = 1e-9,
    mu: float = 1.0
) -> torch.Tensor:
    """
    Compute per-scenario utility U(x; env^{(s)}) for L environment samples.

    Args
    ----
    p_samples : torch.Tensor shape (L, M)
        popularity vectors for each scenario (rows sum to 1)
    lam_samples : torch.Tensor shape (L,)  OR (L,1)
        user density samples (one scalar per scenario)
    x : torch.Tensor shape (M,) or (1, M)
        caching marginal vector (same for all scenarios)
    theta : float
        SINR threshold (2^T - 1)
    lambda_I_factor : float
        fraction (or factor) mapping lam -> lam_I (active interferer intensity)
        default 1.0 (lambda_I = lambda)
    P_t, N0, mu : floats model params

    Returns
    -------
    U : torch.Tensor shape (L,)
        utility for each scenario
    """
    # normalize shapes
    if x.dim() == 1:
        x_vec = x.view(1, -1)  # (1, M)
    else:
        x_vec = x

    device = p_samples.device
    dtype = p_samples.dtype

    # lam_samples shape (L,) -> (L,1)
    lam = lam_samples.view(-1, 1).to(dtype=dtype, device=device)  # (L,1)
    # compute lam_i = lam * x (broadcast)
    lam_i = lam * x_vec.to(dtype=dtype, device=device)  # (L,M)
    # interfering intensity lam_I (per-scenario)
    lam_I = lam.view(-1) * float(lambda_I_factor)  # (L,)
    lam_I = lam_I.view(-1, 1)  # broadcastable to (L,M)

    # compute per-file success probabilities (L,M)
    P_succ = p_succ_alpha4(lam_i, lam_I, theta, P_t=P_t, N0=N0, mu=mu)

    # per-scenario utility U = sum_i p_i * P_succ_i
    U = torch.sum(p_samples * P_succ, dim=1)  # (L,)
    # ensure numerical safety
    U = torch.clamp(U, 0.0, 1.0)
    return U


# ----------------------------
# Smoothed CVaR loss (minimize)
# ----------------------------
def _sigma_tau(z: torch.Tensor, tau: float) -> torch.Tensor:
    """
    derivative of s_tau(z) wrt z: sigma_tau(z) = exp(tau z)/(1 + exp(tau z)) = sigmoid(tau z)
    """
    return torch.sigmoid(tau * z)

def solve_t_star_bisection(
    ell: torch.Tensor,
    gamma: float,
    tau: float,
    tol: float = 1e-4,
    max_iter: int = 40
) -> torch.Tensor:
    """
    Solve for t* that satisfies: 1 = (1/(gamma L)) * sum_{s} sigma_tau(ell_s - t)
    using bisection. Input ell is shape (L,) or (B, L) for batch; returns t_star (scalar or (B,)).
    """
    single = (ell.dim() == 1)
    if single:
        ell_b = ell.unsqueeze(0)  # shape (1,L)
    else:
        ell_b = ell  # shape (B,L)

    B, L = ell_b.shape
    device = ell_b.device
    dtype = ell_b.dtype

    # bounds: t in [min(ell)-R, max(ell)+R] with R margin
    ell_min, ell_max = ell_b.min(dim=1).values, ell_b.max(dim=1).values
    R = (ell_max - ell_min).clamp(min=1.0)  # margin at least 1
    t_lo = (ell_min - R).clone()
    t_hi = (ell_max + R).clone()

    # ensure shapes (B,)
    t_lo = t_lo.to(device=device, dtype=dtype)
    t_hi = t_hi.to(device=device, dtype=dtype)

    def f_val(t_vec):
        # t_vec shape (B,)
        # compute 1 - (1/(gamma L)) * sum sigma_tau(ell - t)
        t_exp = t_vec.view(-1, 1)
        s = _sigma_tau(ell_b - t_exp, tau).sum(dim=1)  # (B,)
        return 1.0 - (1.0 / (gamma * float(L))) * s

    f_lo = f_val(t_lo)
    f_hi = f_val(t_hi)

    # Adjust if endpoints do not bracket zero: expand interval
    expand_iter = 0
    while torch.all((f_lo * f_hi) > 0) and (expand_iter < 10):
        # expand both sides
        t_lo = t_lo - (2.0 ** expand_iter)
        t_hi = t_hi + (2.0 ** expand_iter)
        f_lo = f_val(t_lo)
        f_hi = f_val(t_hi)
        expand_iter += 1

    # bisection loop (vectorized): track t_lo and t_hi per batch element
    for _ in range(max_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        f_mid = f_val(t_mid)
        # We'll use monotonic properties numerically by checking f_lo * f_mid <= 0
        cond = (f_lo * f_mid <= 0)
        # update intervals per element
        t_hi = torch.where(cond, t_mid, t_hi)
        f_hi = torch.where(cond, f_hi, f_mid)
        t_lo = torch.where(cond, t_lo, t_mid)
        f_lo = torch.where(cond, f_lo, f_mid)

        if torch.max(torch.abs(t_hi - t_lo)) < tol:
            break

    t_star = 0.5 * (t_lo + t_hi)
    return t_star.squeeze(0) if single else t_star

def smoothed_cvar_loss_from_utilities(
    U_batch: torch.Tensor,
    gamma: float,
    tau: float,
    minimize: bool = True,
    per_sample: bool = False,
    tol: float = 1e-4,
    max_iter: int = 40
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute smoothed CVaR of the *loss* ell = 1 - U for each minibatch element,
    solve inner t* per minibatch element (vectorized bisection), and return
    the mean CVaR value (scalar) and per-sample CVaR (shape B,).
    """
    single = (U_batch.dim() == 1)
    if single:
        U = U_batch.unsqueeze(0)  # (1, L)
    else:
        U = U_batch  # (B, L)
    B, L = U.shape

    # loss ell = 1 - U (we minimize loss)
    ell = 1.0 - U  # (B, L)

    # solve t* per row
    t_star = solve_t_star_bisection(ell, gamma=gamma, tau=tau, tol=tol, max_iter=max_iter)  # shape (B,)

    # CRITICAL FIX: Use F.softplus(z, beta=tau) which correctly implements
    # (1/tau) * log(1 + exp(tau * z)).
    t_star_col = t_star.view(-1, 1)
    s_vals = F.softplus(ell - t_star_col, beta=tau)  # (B, L)
    
    # compute Phi(t_star) = t + (1/(gamma L)) sum s_tau(ell - t)
    Phi = t_star + (1.0 / (gamma * float(L))) * s_vals.sum(dim=1)  # shape (B,)

    loss_mean = Phi.mean()
    return (loss_mean, Phi.squeeze(0)) if per_sample else (loss_mean, None)

if __name__ == "__main__":
    pass