# src/train.py
"""
Training script for Neural L2O CVaR-based D2D caching.

Main features:
 - Uncertainty injection: for each measurement q (batch entry) we sample L
   posterior scenarios (p, lambda) from an offline env_pool and evaluate utility
   across those L samples to form the empirical CVaR loss.
 - Smoothed CVaR: we use a softplus smoothing of the hinge to obtain a smooth,
   differentiable surrogate and compute the inner argmax (t^*) per minibatch
   with a cheap 1-D bisection solve. We treat t^* as fixed for backprop (envelope theorem).
 - Projection: network outputs raw scores y in R^M; we project to feasible
   caching marginals x in the capped simplex {x in [0,1]^M : sum x <= S}.
 - Warm start (optional): pretrain to posterior-mean optimal x (MSE regression)
   to stabilize subsequent CVaR training.
 - Checkpointing, validation evaluation (mean & CVaR metrics), gradient clipping.

Typical workflow:
    env_pool = list(build_env_pool(...))   # offline (N_pool pairs)
    model, optimizer = create_mlp(...)
    train_loop(model, optimizer, env_pool, dataset, ...)
"""

from typing import Sequence, Tuple, Optional
import os
import time
import math
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Project utils assumed in losses.py
# from src.losses import project_capped_simplex, softplus_scaled
# from src.model import create_mlp
# from src.env_pool import build_env_pool
# from src.eval import evaluate_policy_batch  # optional

# If your project structure differs, adapt the imports above accordingly.


# -------------------------
# Helpers and small utils
# -------------------------
def softplus_scaled(z: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Softplus surrogate s_tau(z) = (1/tau) * log(1 + exp(tau * z)).
    Implement using torch.nn.functional.softplus with beta=tau.
    Note: torch.nn.functional.softplus(x, beta) = (1/beta) * log(1 + exp(beta*x))
    """
    # ensure tau > 0
    return F.softplus(z, beta=max(tau, 1e-12))


def sigmoid_tau(z: torch.Tensor, tau: float) -> torch.Tensor:
    """Derivative of softplus_scaled: sigma_tau(z) = sigmoid(tau * z)."""
    return torch.sigmoid(tau * z)


def find_t_star_smoothed(u: torch.Tensor, gamma: float, tau: float,
                        max_iter: int = 60, tol: float = 1e-4) -> torch.Tensor:
    """
    Find t^* = argmax_t [ t - (1/(gamma L)) sum_s s_tau(t - u_s) ] via bisection.
    We solve the first-order condition:  1 - (1/(gamma L)) sum_s sigma_tau(t - u_s) = 0.

    Inputs:
      u: (L,) tensor of utilities (on device)
      gamma: tail probability (0<gamma<1)
      tau: softplus beta parameter (>0). Larger tau -> sharper approximation.
      max_iter, tol: bisection stopping criteria.

    Returns:
      scalar torch.Tensor t_star (detached; does not require grad)
    """
    # ensure u is 1D tensor
    if u.dim() != 1:
        u = u.view(-1)

    L = float(u.shape[0])
    # bracket: t in [min(u) - R, max(u) + R]
    umin, umax = u.min().item(), u.max().item()
    R = max(1.0, 0.1 * (umax - umin + 1e-6))
    left = umin - R
    right = umax + R

    # convert to tensors on same device dtype
    device = u.device
    left_t = torch.tensor(left, device=device)
    right_t = torch.tensor(right, device=device)

    # bisection on scalar t: evaluate g(t) = 1 - (1/(gamma L)) sum sigma_tau(t - u)
    def g(t: torch.Tensor) -> torch.Tensor:
        # t is scalar tensor
        vals = sigmoid_tau(t - u, tau)  # (L,)
        return 1.0 - (1.0 / (gamma * L)) * torch.sum(vals)

    # Ensure sign difference
    g_left = g(left_t).item()
    g_right = g(right_t).item()
    # if g_left * g_right > 0, expand bracket (rare)
    expand = 0
    while g_left * g_right > 0 and expand < 10:
        expand += 1
        R *= 2.0
        left_t = torch.tensor(umin - R, device=device)
        right_t = torch.tensor(umax + R, device=device)
        g_left = g(left_t).item()
        g_right = g(right_t).item()

    # Bisection loop
    for _ in range(max_iter):
        mid = 0.5 * (left_t + right_t)
        val = g(mid)
        if abs(val.item()) < tol:
            break
        # choose side where root lies
        if g_left * val.item() <= 0:
            right_t = mid
            g_right = val.item()
        else:
            left_t = mid
            g_left = val.item()

    t_star = 0.5 * (left_t + right_t)
    # detach to stop gradients through inner solver (envelope theorem)
    return t_star.detach()


# -------------------------
# Training loop
# -------------------------
@dataclass
class TrainConfig:
    # data / env pool
    N_pool: int = 2000          # size of env_pool (offline)
    L: int = 500                # number of posterior draws per measurement
    batch_size: int = 16
    gamma_tail: float = 0.05    # CVaR level (e.g., 0.05)
    tau: float = 20.0           # softplus sharpness (recommend 5-20). larger->closer to hinge
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    validate_every: int = 1     # run validation every N epochs
    warmstart_epochs: int = 0   # number of pretrain epochs to regress to posterior-mean optimum
    verbose: bool = True


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    env_pool: Sequence[Tuple[np.ndarray, float]],
    dataset_q: np.ndarray,
    S: float,
    compute_utility_fn,
    project_fn,
    config: TrainConfig,
    val_dataset_q: Optional[np.ndarray] = None,
    eval_fn=None
) -> None:
    """
    Main training loop.

    Args:
      model: PyTorch model mapping q -> raw scores y (shape (batch, M))
      optimizer: optimizer for model parameters
      env_pool: list-like of (p_vector (np.array length M), lambda_scalar) pairs
      dataset_q: numpy array of measurements q (shape [N_measurements, input_dim])
      S: cache capacity (expected). Projection target sum <= S.
      compute_utility_fn: callable (p:torch.Tensor, lam:torch.Tensor, x:torch.Tensor) -> utility scalar
                          Must accept p shape (M,), lam scalar tensor, x shape (M,) and return scalar tensor
      project_fn: callable project_capped_simplex(y: torch.Tensor, S: float) -> x:torch.Tensor
      config: TrainConfig dataclass
      val_dataset_q: optional validation dataset (numpy)
      eval_fn: optional evaluation function used for validation (model, val_dataset, env_pool, S) -> metrics dict

    Returns:
      None (saves checkpoints in config.checkpoint_dir)
    """
    device = torch.device(config.device)
    model = model.to(device)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    N = dataset_q.shape[0]
    indices = np.arange(N) # make np array 0 to n-1

    # convert env_pool to in-memory numpy arrays for fast indexing
    env_pool_list = list(env_pool)
    pool_size = len(env_pool_list)
    rng = np.random.default_rng(seed=12345)

    # helper to sample L indices
    def sample_env_indices(L):
        return rng.integers(0, pool_size, size=L) 
        # maryam : make sure that samples are not the same => use replace=False

    # training
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        model.train()

        # shuffle dataset indices
        rng.shuffle(indices)

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, N, config.batch_size):
            batch_idx = indices[start:start + config.batch_size]
            batch_q = dataset_q[batch_idx]  # shape (B, input_dim)
            B_actual = batch_q.shape[0]

            optimizer.zero_grad()
            J_sum = torch.tensor(0.0, device=device)  # accumulate J_b (CVaR objective) over batch

            # Process each measurement q in the minibatch (can be vectorized further)
            for q_np in batch_q:
                # 1) forward through model -> raw scores y, project to feasible x
                q_t = torch.from_numpy(q_np).float().to(device).unsqueeze(0)  # (1, in_dim)
                y = model(q_t).squeeze(0)  # (M,)
                x = project_fn(y, S)      # returns (M,) on same device

                # 2) sample L posterior scenarios from env_pool
                inds = sample_env_indices(config.L)
                u_list = []
                for s in inds:
                    p_s, lam_s = env_pool_list[0][int(s)],env_pool_list[1][int(s)]
                    # convert to torch tensors on device
                    p_t = torch.from_numpy(np.asarray(p_s, dtype=np.float32)).to(device)  # (M,)
                    lam_t = torch.tensor(float(lam_s), dtype=torch.float32, device=device)
                    # 3) compute per-scenario utility (scalar)
                    u_s = compute_utility_fn(p_t, lam_t, x)  # must return torch scalar
                    # ensure scalar tensor
                    u_list.append(u_s)

                u = torch.stack(u_list, dim=0).view(-1)  # (L,)

                # 4) inner solve for t* (smoothed CVaR)
                t_star = find_t_star_smoothed(u, gamma=config.gamma_tail, tau=config.tau)
                # compute surrogate objective value J_b = t_star - (1/(gamma L)) sum_s s_tau(t_star - u_s)
                # Use softplus_scaled (handles beta=tau)
                svals = softplus_scaled(t_star - u, config.tau)  # (L,)
                J_b = t_star - (1.0 / (config.gamma_tail * float(u.shape[0]))) * torch.sum(svals)

                # Treat t_star as fixed (we detached in solver), so J_b graph flows only w.r.t. u -> Theta
                J_sum = J_sum + J_b

            # average over batch and convert to minimization loss
            J_batch_mean = J_sum / float(B_actual)
            loss = -J_batch_mean  # we maximize CVaR; optimizer minimizes -CVaR
            loss.backward()

            # gradient clipping
            if config.clip_grad_norm is not None and config.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # epoch stats
        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(1, num_batches)

        if config.verbose:
            print(f"[Epoch {epoch:03d}] loss={avg_loss:.6f} time={epoch_time:.1f}s")

        # validation / checkpoint
        if (val_dataset_q is not None) and (epoch % config.validate_every == 0):
            model.eval()
            if eval_fn is not None:
                metrics = eval_fn(model, val_dataset_q, env_pool_list, project_fn, compute_utility_fn, S,
                                  gamma=config.gamma_tail, device=device)
                print(f"  Validation: {metrics}")
            # save checkpoint
            ckpt_path = os.path.join(config.checkpoint_dir, f"ckpt_epoch{epoch:03d}.pt")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()}, ckpt_path)

    # final save
    final_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    torch.save({'epoch': config.epochs, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()}, final_path)
    print(f"Training finished. Final model saved to {final_path}")


# -------------------------
# Example evaluation function
# -------------------------
def evaluate_model_simple(model, val_dataset_q, env_pool_list, project_fn, compute_utility_fn, S,
                          gamma=0.05, device=torch.device("cpu"), n_env_eval=200):
    """
    Simple evaluation: for each q in val_dataset, draw n_env_eval scenario draws
    and compute the realized utility distribution under the policy x = Project(model(q)).
    Returns aggregated metrics: mean utility, CVaR_gamma, VaR_gamma across validation set.

    NOTE: This is intentionally simple; for publication-quality experiments you will
    vectorize and compute larger-sample estimates.
    """
    model = model.to(device)
    model.eval()

    results = []
    rng = np.random.default_rng(0)
    pool_size = len(env_pool_list)
    for q_np in val_dataset_q:
        with torch.no_grad():
            q_t = torch.from_numpy(q_np).float().to(device).unsqueeze(0)
            y = model(q_t).squeeze(0)
            x = project_fn(y, S)

        # draw n_env_eval scenarios
        inds = rng.integers(0, pool_size, size=n_env_eval)
        u_vals = []
        for s in inds:
            p_s, lam_s = env_pool_list[int(s)]
            p_t = torch.from_numpy(np.asarray(p_s, dtype=np.float32)).to(device)
            lam_t = torch.tensor(float(lam_s), dtype=torch.float32, device=device)
            u = compute_utility_fn(p_t, lam_t, x)
            u_vals.append(u.item())

        u_arr = np.array(u_vals)
        mean_u = float(u_arr.mean())
        # VaR: empirical gamma-quantile, CVaR: mean of worst gamma fraction
        var_gamma = float(np.quantile(u_arr, gamma))
        k = max(1, int(np.ceil(gamma * len(u_arr))))
        cvar_gamma = float(np.mean(np.sort(u_arr)[:k]))
        results.append((mean_u, var_gamma, cvar_gamma))

    # aggregate across validation set
    arr = np.array(results)
    metrics = {
        'mean_utility_mean': float(arr[:, 0].mean()),
        'mean_utility_std': float(arr[:, 0].std()),
        f'VaR_{gamma}': float(arr[:, 1].mean()),
        f'CVaR_{gamma}': float(arr[:, 2].mean())
    }
    return metrics


# -------------------------
# If called as script: small smoke-run example (toy)
# -------------------------
if __name__ == "__main__":
    # Minimal smoke test (toy sizes)
    from src.model import create_mlp
    from src.env_pool import build_env_pool
    from src.losses import project_capped_simplex
    # toy compute_utility: simple dot product (replace with your real utility)
    def toy_utility(p, lam, x):
        # p: (M,), x: (M,)
        return torch.sum(p * (1.0 - 0.5 * x))  # dummy

    M = 20
    in_extra = 1
    model, optimizer = create_mlp(M=M, in_extra=in_extra, lr=1e-3, weight_decay=1e-4)
    # make toy env_pool
    env_pool = list(build_env_pool(N_pool=200, M=M, W=100, gamma_r=1.0, a0=1, b0=1, A_user=10, lambda_=2.0))
    # toy dataset of q: here we use p concatenated with one lambda measurement (just demo)
    dataset_q = np.vstack([np.concatenate([p, np.array([lam])]) for (p, lam) in env_pool[:100]])
    config = TrainConfig(N_pool=200, L=64, batch_size=8, gamma_tail=0.05, tau=8.0, epochs=2,
                         lr=1e-3, device="cpu", checkpoint_dir="./checkpoints_toy", verbose=True)

    train_loop(model, optimizer, env_pool, dataset_q, S=5.0, compute_utility_fn=toy_utility,
               project_fn=project_capped_simplex, config=config,
               val_dataset_q=dataset_q[:20],
               eval_fn=evaluate_model_simple)
