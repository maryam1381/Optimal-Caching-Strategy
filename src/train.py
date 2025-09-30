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

from pyparsing import Callable
import torch
import torch.nn.functional as F
from src.eval import mean_opt_plug_in, popularity_deterministic_topS
from src.model_eval import evaluate_all_policies
from src.utils import to_torch, append_row_csv as append_log

# Project utils assumed in losses.py
from src.losses import smoothed_cvar_loss_from_utilities, project_capped_simplex
# from src.model import create_mlp
# from src.env_pool import build_env_pool
# from src.eval import evaluate_policy_batch  # optional

# If your project structure differs, adapt the imports above accordingly.


# -------------------------
# Helpers and small utils
# -------------------------
# Removed unused sigmoid_tau helper.


# -------------------------
# Training loop
# -------------------------
# @dataclass
# class TrainConfig:
#     # data / env pool
#     N_pool: int = 2000          # size of env_pool (offline)
#     L: int = 500                # number of posterior draws per measurement
#     batch_size: int = 16
#     gamma_tail: float = 0.05    # CVaR level (e.g., 0.05)
#     tau: float = 20.0           # softplus sharpness (CRITICAL FIX: Changed from 5 to 20)
#     epochs: int = 100
#     lr: float = 1e-3
#     weight_decay: float = 1e-4
#     clip_grad_norm: float = 1.0
#     device: str = "cpu"
#     checkpoint_dir: str = "./checkpoints"
#     validate_every: int = 1     # run validation every N epochs
#     warmstart_epochs: int = 0   # number of pretrain epochs to regress to posterior-mean optimum
#     verbose: bool = True
#     sample_without_replacement: bool = True

#     zipf_exponent: float = 0.8
#     lambda_true: float = 2.5
#     W: int = 200
#     seed: int = 42


# def train_loop(
#     model: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     env_pool: Tuple[np.ndarray, np.ndarray],
#     dataset_q: np.ndarray,
#     S: float,
#     compute_utility_fn,
#     project_fn,
#     config: TrainConfig,
#     rng: np.random.Generator,
#     val_dataset_q: Optional[np.ndarray] = None,
#     eval_fn=None,
# ) -> None:
#     """
#     Main training loop.

#     Args:
#       model: PyTorch model mapping q -> raw scores y (shape (batch, M))
#       optimizer: optimizer for model parameters
#       env_pool: tuple of numpy arrays (P_pool, LAM_pool)
#       dataset_q: numpy array of measurements q (shape [N_measurements, input_dim])
#       S: cache capacity (expected). Projection target sum <= S.
#       compute_utility_fn: callable (p:torch.Tensor, lam:torch.Tensor, x:torch.Tensor) -> utility scalar
#                           Must accept p shape (M,), lam scalar tensor, x shape (M,) and return scalar tensor
#       project_fn: callable project_capped_simplex(y: torch.Tensor, S: float) -> x:torch.Tensor
#       config: TrainConfig dataclass
#       rng: numpy.random.Generator instance for reproducibility
#       val_dataset_q: optional validation dataset (numpy)
#       eval_fn: optional evaluation function used for validation (model, val_dataset, env_pool, S) -> metrics dict

#     Returns:
#       None (saves checkpoints in config.checkpoint_dir)
#     """
#     device = torch.device(config.device)
#     model.to(device)
#     os.makedirs(config.checkpoint_dir, exist_ok=True)

#     N = dataset_q.shape[0]
#     idx_all = np.arange(N)

#     # --- stack env pool to arrays for fast gather ---
#     p_pool, lam_pool= env_pool
#     N_pool = len(p_pool)
#     assert N_pool > 0, "env_pool is empty"

#     # sampling helper
#     def sample_env_indices(B: int, L: int) -> np.ndarray:
#         if config.sample_without_replacement and L <= N_pool:
#             # different set per row; still vectorized
#             inds = np.vstack([rng.choice(N_pool, size=L, replace=False) for _ in range(B)])
#         else:
#             inds = rng.integers(0, N_pool, size=(B, L))
#         return inds
    
#     log_filename = (
#         f"TRAIN_Zipf{config.zipf_exponent:.1f}_"
#         f"Lam{config.lambda_true:.1f}_"
#         f"W{config.W}_"
#         f"L{config.L}_"
#         f"Seed{config.seed}.csv"
#     )
#     training_log_path = os.path.join(config.checkpoint_dir, log_filename)

#     p_bar_global = p_pool.mean(axis=0)
#     lambda_bar_global = float(lam_pool.mean())
#     M = p_bar_global.shape[0]

#     for epoch in range(1, config.epochs + 1):
#         t0 = time.time()
#         model.train()
#         rng.shuffle(idx_all)

#         epoch_loss = 0.0
#         num_batches = 0

#         for start in range(0, N, config.batch_size):
#             sel = idx_all[start:start + config.batch_size]
#             q_batch = dataset_q[sel]                        # (B, in_dim)
#             B = q_batch.shape[0]

#             optimizer.zero_grad()

#             # (1) Forward entire batch → y → project to x
#             q_t = torch.from_numpy(q_batch).float().to(device)     # (B, in_dim)
#             y = model(q_t,)

#             x_rows = [project_fn(y[b], S) for b in range(B)]
#             x = torch.stack(x_rows, dim=0)                         # (B, M)

#             # (2) Sample L envs per batch row, gather p, lambda
#             inds = sample_env_indices(B, config.L)                 # (B, L)
#             p_batch_np   = p_pool[inds]                            # (B, L, M)
#             lam_batch_np = lam_pool[inds]                          # (B, L)

#             p_batch   = to_torch(p_batch_np,   device)  # (B, L, M)
#             lam_batch = to_torch(lam_batch_np,   device)   # (B, L)
#             x_exp     = x[:, None, :]                              # (B, 1, M) for broadcast

#             # (3) Compute utilities for all (B,L) scenarios
#             # Preferred: compute_utility_fn supports vectorized inputs and returns (B,L)
#             U = None
#             try:
#                 U = compute_utility_fn(p_batch, lam_batch, x_exp)  # expect (B, L)
#                 if not (torch.is_tensor(U) and U.shape == (B, config.L)):
#                     U = None
#             except Exception:
#                 U = None

#             if U is None:
#                 # Fallback: loop over L or (B,L) with minimal Python overhead
#                 U_list = []
#                 for l in range(config.L):
#                     p_l   = p_batch[:, l, :]                 # (B, M)
#                     lam_l = lam_batch[:, l]                  # (B,)
#                     try:
#                         # allow a (B,M),(B,) path
#                         U_l = compute_utility_fn(p_l, lam_l, x)    # (B,)
#                         if not (torch.is_tensor(U_l) and U_l.shape == (B,)):
#                             raise RuntimeError
#                     except Exception:
#                         # final fallback: loop across B
#                         u_rows = []
#                         for b in range(B):
#                             u_rows.append(compute_utility_fn(p_l[b], lam_l[b], x[b]))
#                         U_l = torch.stack(u_rows, dim=0)
#                     U_list.append(U_l)
#                 U = torch.stack(U_list, dim=1)               # (B, L)

#             # (4) Compute CVaR loss from utilities
#             loss, _ = smoothed_cvar_loss_from_utilities(U, gamma=config.gamma_tail, tau=config.tau)

#             # (5) Backprop, clip, step
#             loss.backward()
#             if config.clip_grad_norm and config.clip_grad_norm > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
#             optimizer.step()

#             epoch_loss += float(loss.detach().cpu())
#             num_batches += 1

#         if config.verbose:
#             print(f"[Epoch {epoch:03d}] loss={epoch_loss/max(1,num_batches):.6f} "
#                   f"time={time.time()-t0:.1f}s")

#         # Optional validation + checkpoint
#         if (val_dataset_q is not None) and (epoch % config.validate_every == 0):
#             model.eval()
#             if eval_fn is not None:
#                 # Pass a list of tuples to the evaluation function as it expects it
#                 env_pool_list = list(zip(p_pool, lam_pool))
#                 metrics = eval_fn(
#                     model, val_dataset_q, env_pool_list, project_fn, compute_utility_fn, S,
#                     gamma=config.gamma_tail, device=device, rng=rng
#                 )
#                 print("  Validation:", metrics)
#                 log_row = {
#                     'epoch': epoch,
#                     'loss_train_mean': epoch_loss/max(1,num_batches),
#                     'gamma_tail': config.gamma_tail,
#                     'tau': config.tau,
#                     **metrics, # Include all calculated validation metrics
#                     'S_cache': S,
#                     # Add current experiment parameters for tracking
#                 }
#                 append_log(training_log_path, log_row)
            
#             # Checkpoint save
#             ckpt = {
#                 "epoch": epoch,
#                 "model_state": model.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#                 "config": config.__dict__,
#             }
#             os.makedirs(config.checkpoint_dir, exist_ok=True)
#             torch.save(ckpt, os.path.join(config.checkpoint_dir, f"ckpt_epoch_{epoch:03d}.pt"))

#     # final save
#     torch.save({"epoch": config.epochs, "model_state": model.state_dict()},
#                os.path.join(config.checkpoint_dir, "final_model.pt"))


def _batch_utility_from_env(
    p_batch: torch.Tensor,      # (B, L, M)
    lam_batch: torch.Tensor,    # (B, L)
    x: torch.Tensor,            # (B, M)
    compute_utility_fn: Callable
) -> torch.Tensor:
    """
    Vectorized wrapper over a per-sample utility function that expects (L,M), (L,), (M,)
    and returns (L,). We flatten the (B,L) grid to (B*L), call once, then reshape back to (B,L).
    Keeps the computation fully differentiable w.r.t. x.

    Returns:
        U: torch.Tensor with shape (B, L)
    """
    assert p_batch.dim() == 3, f"p_batch must be (B,L,M), got {tuple(p_batch.shape)}"
    assert lam_batch.dim() == 2, f"lam_batch must be (B,L), got {tuple(lam_batch.shape)}"
    assert x.dim() == 2, f"x must be (B,M), got {tuple(x.shape)}"

    B, L, M = p_batch.shape
    assert lam_batch.shape == (B, L)

    # Flatten (B,L,M)->(B*L,M); (B,L)->(B*L,)
    p_flat   = p_batch.reshape(B * L, M).contiguous()
    lam_flat = lam_batch.reshape(B * L).contiguous()
    # Repeat each x[b] for its L scenarios -> (B*L, M)
    x_flat   = x.unsqueeze(1).expand(B, L, M).reshape(B * L, M).contiguous()

    # Call the user utility; expected output is (B*L,) or (B*L,1)
    U_flat = compute_utility_fn(p_flat, lam_flat, x_flat)

    if not torch.is_tensor(U_flat):
        raise TypeError("compute_utility_fn must return a torch.Tensor")

    if U_flat.dim() == 2 and U_flat.shape[1] == 1:
        U_flat = U_flat.view(-1)

    if U_flat.dim() != 1 or U_flat.numel() != B * L:
        raise RuntimeError(
            f"utility_from_env_samples must return shape (B*L,), got {tuple(U_flat.shape)}"
        )

    U = U_flat.view(B, L)
    return torch.clamp(U, 0.0, 1.0)



@dataclass
class TrainConfig:
    N_pool: int = 2000
    L: int = 500
    batch_size: int = 16
    gamma_tail: float = 0.05
    tau: float = 20.0
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    validate_every: int = 1
    verbose: bool = True
    sample_without_replacement: bool = True
    zipf_exponent: float = 0.8
    lambda_true: float = 2.5
    W: int = 200
    seed: int = 42

def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    env_pool: Tuple[np.ndarray, np.ndarray],
    dataset_q: np.ndarray,
    S: float,
    compute_utility_fn: Callable,
    project_fn: Callable,
    config: TrainConfig,
    rng: np.random.Generator,
    val_dataset_q: Optional[np.ndarray] = None,
) -> None:
    """
    The complete training loop with batch processing, validation, 
    comparative logging, and checkpointing.
    """
    device = torch.device(config.device)
    model.to(device)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    N = dataset_q.shape[0]
    idx_all = np.arange(N)

    p_pool, lam_pool = env_pool
    N_pool = len(p_pool)
    assert N_pool > 0, "env_pool is empty"

    log_filename = (
        f"TRAIN_LOG_Compare_Zipf{config.zipf_exponent:.1f}_"
        f"L{config.L}_Seed{config.seed}.csv"
    )
    training_log_path = os.path.join(config.checkpoint_dir, log_filename)
    print(f"Logging training and comparison progress to: {training_log_path}")

    # --- Pre-calculate global stats for baselines (do it once) ---
    p_bar_global = p_pool.mean(axis=0)
    lambda_bar_global = float(lam_pool.mean())
    M = p_bar_global.shape[0]
    
    # --- Sampling helper ---
    def sample_env_indices(B: int, L: int) -> np.ndarray:
        if config.sample_without_replacement and L <= N_pool:
            inds = np.vstack([rng.choice(N_pool, size=L, replace=False) for _ in range(B)])
        else:
            inds = rng.integers(0, N_pool, size=(B, L))
        return inds

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        model.train()
        rng.shuffle(idx_all)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for start in range(0, N, config.batch_size):
            sel = idx_all[start:start + config.batch_size]
            q_batch = dataset_q[sel]                        # (B, in_dim)
            B = q_batch.shape[0]

            optimizer.zero_grad()

            # (1) Forward pass: q -> y -> x (policy)
            q_t = to_torch(q_batch, device)
            y = model(q_t)                                  # (B, M)
            # Sanity: model output matches pool's M
            if y.shape[1] != M:
                raise RuntimeError(f"Model output dim {y.shape[1]} != M {M}")

            x_rows = [project_fn(y[b], S) for b in range(B)]
            x = torch.stack(x_rows, dim=0)                  # (B, M)

            # (2) Sample L environment scenarios per batch item
            inds = sample_env_indices(B, config.L)          # (B, L)
            p_batch_np = p_pool[inds]                       # (B, L, M)
            lam_batch_np = lam_pool[inds]                   # (B, L)

            p_batch = to_torch(p_batch_np, device)
            lam_batch = to_torch(lam_batch_np, device)

            # (3) Compute utilities across all (B, L) scenarios
            #     Use the batch-aware wrapper to guarantee shape (B, L)
            U = _batch_utility_from_env(p_batch, lam_batch, x, compute_utility_fn)  # (B, L)
            if U.shape != (B, config.L):
                raise RuntimeError(f"Utility must be (B,L). Got {tuple(U.shape)}")

            # (4) Compute the CVaR loss from the utility samples
            loss, _ = smoothed_cvar_loss_from_utilities(U, gamma=config.gamma_tail, tau=config.tau)

            # (5) Backpropagate, clip gradients, and update weights
            loss.backward()
            if config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        print(f"[Epoch {epoch:03d}] Train Loss={avg_epoch_loss:.6f} | Time={time.time()-t0:.1f}s")

        # --- VALIDATION AND LOGGING ---
        if (val_dataset_q is not None) and (epoch % config.validate_every == 0):
            print("  Running validation for all policies...")
            model.eval()

            # 1. Define the baseline policy functions
            mean_opt_policy = lambda q_np: mean_opt_plug_in(
                p_bar=p_bar_global, lambda_bar=lambda_bar_global, S=S,
                compute_utility_fn=compute_utility_fn, project_fn=project_capped_simplex, M=M, device=config.device
            )
            popularity_policy = lambda q_np: popularity_deterministic_topS(p_bar_global, S)

            # 2. Create the dictionary of all policies to compare
            policies_to_evaluate = {
                "RLO_CVaR": model,
                "Mean-Opt": mean_opt_policy,
                "Popularity-TopS": popularity_policy
            }

            # 3. Call the multi-policy evaluator
            env_pool_list = list(zip(p_pool, lam_pool))
            all_metrics = evaluate_all_policies(
                policies_to_evaluate=policies_to_evaluate,
                val_dataset_q=val_dataset_q,
                env_pool_list=env_pool_list,
                project_fn=project_capped_simplex,
                compute_utility_fn=compute_utility_fn,
                S=S,
                gamma=config.gamma_tail,
                device=device,
                rng=rng
            )

            # 4. Flatten the nested metrics dictionary for logging
            flat_metrics = {}
            for policy_name, metrics_dict in all_metrics.items():
                for metric_name, value in metrics_dict.items():
                    flat_metrics[f"{policy_name}_{metric_name}"] = value
            
            print(f"  Validation Metrics (flat): {flat_metrics}")

            # 5. Prepare and append the log row
            log_row = {
                'epoch': epoch,
                'train_loss_mean': avg_epoch_loss,
                **flat_metrics,
            }
            append_log(training_log_path, log_row)

            # --- CHECKPOINT SAVE ---
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config.__dict__,
            }
            torch.save(ckpt, os.path.join(config.checkpoint_dir, f"ckpt_epoch_{epoch:03d}.pt"))

    # --- FINAL MODEL SAVE ---
    torch.save({
        "epoch": config.epochs, 
        "model_state": model.state_dict()
    }, os.path.join(config.checkpoint_dir, "final_model.pt"))
    print("Training complete.")

# -------------------------
# Example evaluation function (updated for loss-based metrics)
# -------------------------
def evaluate_model_simple(model, val_dataset_q, env_pool_list, project_fn, compute_utility_fn, S,
                          gamma=0.05, device=torch.device("cpu"), n_env_eval=200, rng=np.random.default_rng()):
    """
    Simple evaluation: for each q in val_dataset, draw n_env_eval scenario draws
    and compute the realized utility distribution under the policy x = Project(model(q)).
    Returns aggregated metrics: mean utility, mean loss, VaR_gamma(loss), CVaR_gamma(loss) across validation set.
    """
    model = model.to(device)
    model.eval()

    results = []
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
        
        per_sample_loss = 1.0 - u_arr
        
        mean_u = float(u_arr.mean())
        mean_loss = float(per_sample_loss.mean())
        
        # VaR on loss: empirical gamma-quantile of loss
        var_loss_gamma = float(np.quantile(per_sample_loss, gamma)) 
        
        # CVaR on loss: mean of worst gamma fraction of losses.
        k = max(1, int(np.ceil(gamma * len(u_arr))))
        sorted_losses = np.sort(per_sample_loss)
        cvar_loss_gamma = float(np.mean(sorted_losses[-k:]))
        
        results.append((mean_u, mean_loss, var_loss_gamma, cvar_loss_gamma))

    # aggregate across validation set
    arr = np.array(results)
    metrics = {
        'mean_utility_mean': float(arr[:, 0].mean()),
        'mean_utility_std': float(arr[:, 0].std()),
        'mean_loss_mean': float(arr[:, 1].mean()),
        f'VaR_{gamma}': float(arr[:, 2].mean()), # This is VaR on Loss
        f'CVaR_{gamma}': float(arr[:, 3].mean()), # This is CVaR on Loss
    }
    return metrics


if __name__ == "__main__":
    # Minimal smoke test (toy sizes)
    from model import create_mlp
    from env_pool import build_env_pool_simulated
    from losses import project_capped_simplex
    # toy compute_utility: simple dot product (replace with your real utility)
    def toy_utility(p, lam, x):
        # p: (M,), x: (M,)
        return torch.sum(p * (1.0 - 0.5 * x))  # dummy

    M = 20
    in_extra = 1
    model, optimizer = create_mlp(M=M, in_extra=in_extra, lr=1e-3, weight_decay=1e-4)
    # make toy env_pool
    env_pool = build_env_pool_simulated(N_pool=200, M=M, W=100, zipf_exponent=1.0, a0_lambda=1, b0_lambda=1, a0_p=10, lambda_true=2.0, as_torch=False)
    # toy dataset of q: here we use p concatenated with one lambda measurement (just demo)
    p , lam = env_pool
    dataset_q = np.column_stack((p[:100], lam[:100]))
    config = TrainConfig(N_pool=200, L=64, batch_size=8, gamma_tail=0.05, tau=8.0, epochs=5,
                         lr=1e-3, device="cpu", checkpoint_dir="./checkpoints_toy", verbose=True)
    rng = np.random.default_rng(config.seed)

    train_loop(model, optimizer, env_pool, dataset_q, S=5.0, compute_utility_fn=toy_utility,
               project_fn=project_capped_simplex, config=config, rng=rng,
               val_dataset_q=dataset_q[:20],
               eval_fn=evaluate_model_simple)