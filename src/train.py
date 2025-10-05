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
from src.eval import mean_opt_plug_in, plugin_mean_policy, popularity_deterministic_topS
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
@dataclass
class TrainConfig:
    # data / env pool
    N_pool: int = 2000          # size of env_pool (offline)
    L: int = 500                # number of posterior draws per measurement
    batch_size: int = 16
    gamma_tail: float = 0.05    # CVaR level (e.g., 0.05)
    tau: float = 20.0           # softplus sharpness (CRITICAL FIX: Changed from 5 to 20)
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    validate_every: int = 1     # run validation every N epochs
    warmstart_epochs: int = 0   # number of pretrain epochs to regress to posterior-mean optimum
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
    compute_utility_fn,
    project_fn,
    config: TrainConfig,
    rng: np.random.Generator,
    val_dataset_q: Optional[np.ndarray] = None,
    eval_fn=None,
) -> None:
    """
    Main training loop.

    Args:
      model: PyTorch model mapping q -> raw scores y (shape (batch, M))
      optimizer: optimizer for model parameters
      env_pool: tuple of numpy arrays (P_pool, LAM_pool)
      dataset_q: numpy array of measurements q (shape [N_measurements, input_dim])
      S: cache capacity (expected). Projection target sum <= S.
      compute_utility_fn: callable (p:torch.Tensor, lam:torch.Tensor, x:torch.Tensor) -> utility scalar
                          Must accept p shape (M,), lam scalar tensor, x shape (M,) and return scalar tensor
      project_fn: callable project_capped_simplex(y: torch.Tensor, S: float) -> x:torch.Tensor
      config: TrainConfig dataclass
      rng: numpy.random.Generator instance for reproducibility
      val_dataset_q: optional validation dataset (numpy)
      eval_fn: optional evaluation function used for validation (model, val_dataset, env_pool, S) -> metrics dict

    Returns:
      None (saves checkpoints in config.checkpoint_dir)
    """
    device = torch.device(config.device)
    model.to(device)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    N = dataset_q.shape[0]
    idx_all = np.arange(N)

    # --- stack env pool to arrays for fast gather ---
    p_pool, lam_pool= env_pool
    N_pool = len(p_pool)
    assert N_pool > 0, "env_pool is empty"

    # sampling helper
    def sample_env_indices(B: int, L: int) -> np.ndarray:
        if config.sample_without_replacement and L <= N_pool:
            # different set per row; still vectorized
            inds = np.vstack([rng.choice(N_pool, size=L, replace=False) for _ in range(B)])
        else:
            inds = rng.integers(0, N_pool, size=(B, L))
        return inds
    
    log_filename = (
        f"TRAIN_Zipf{config.zipf_exponent:.1f}_"
        f"Lam{config.lambda_true:.1f}_"
        f"W{config.W}_"
        f"L{config.L}_"
        f"Seed{config.seed}.csv"
    )
    training_log_path = os.path.join(config.checkpoint_dir, log_filename)

    p_bar_global = p_pool.mean(axis=0)
    lambda_bar_global = float(lam_pool.mean())
    M = p_bar_global.shape[0]

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

            # (1) Forward entire batch → y → project to x
            q_t = torch.from_numpy(q_batch).float().to(device)     # (B, in_dim)
            y = model(q_t,)

            x_rows = [project_fn(y[b], S) for b in range(B)]
            x = torch.stack(x_rows, dim=0)                         # (B, M)

            # (2) Sample L envs per batch row, gather p, lambda
            inds = sample_env_indices(B, config.L)                 # (B, L)
            p_batch_np   = p_pool[inds]                            # (B, L, M)
            lam_batch_np = lam_pool[inds]                          # (B, L)

            p_batch   = to_torch(p_batch_np,   device)  # (B, L, M)
            lam_batch = to_torch(lam_batch_np,   device)   # (B, L)
            x_exp     = x[:, None, :]                              # (B, 1, M) for broadcast

            # (3) Compute utilities for all (B,L) scenarios
            # Preferred: compute_utility_fn supports vectorized inputs and returns (B,L)
            U = None
            try:
                U = compute_utility_fn(p_batch, lam_batch, x_exp)  # expect (B, L)
                if not (torch.is_tensor(U) and U.shape == (B, config.L)):
                    U = None
            except Exception:
                U = None

            if U is None:
                # Fallback: loop over L or (B,L) with minimal Python overhead
                U_list = []
                for l in range(config.L):
                    p_l   = p_batch[:, l, :]                 # (B, M)
                    lam_l = lam_batch[:, l]                  # (B,)
                    try:
                        # allow a (B,M),(B,) path
                        U_l = compute_utility_fn(p_l, lam_l, x)    # (B,)
                        if not (torch.is_tensor(U_l) and U_l.shape == (B,)):
                            raise RuntimeError
                    except Exception:
                        # final fallback: loop across B
                        u_rows = []
                        for b in range(B):
                            u_rows.append(compute_utility_fn(p_l[b], lam_l[b], x[b]))
                        U_l = torch.stack(u_rows, dim=0)
                    U_list.append(U_l)
                U = torch.stack(U_list, dim=1)               # (B, L)

            # (4) Compute CVaR loss from utilities
            loss, _ = smoothed_cvar_loss_from_utilities(U, gamma=config.gamma_tail, tau=config.tau)

            # (5) Backprop, clip, step
            loss.backward()
            if config.clip_grad_norm and config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            num_batches += 1

        if config.verbose:
            print(f"[Epoch {epoch:03d}] loss={epoch_loss/max(1,num_batches):.6f} "
                  f"time={time.time()-t0:.1f}s")

        # Optional validation + checkpoint
        # if (val_dataset_q is not None) and (epoch % config.validate_every == 0):
        #     model.eval()
        #     if eval_fn is not None:
        #         # Pass a list of tuples to the evaluation function as it expects it
        #         env_pool_list = list(zip(p_pool, lam_pool))
        #         metrics = eval_fn(
        #             model, val_dataset_q, env_pool_list, project_fn, compute_utility_fn, S,
        #             gamma=config.gamma_tail, device=device, rng=rng
        #         )
        #         print("  Validation:", metrics)
        #         log_row = {
        #             'epoch': epoch,
        #             'loss_train_mean': epoch_loss/max(1,num_batches),
        #             'gamma_tail': config.gamma_tail,
        #             'tau': config.tau,
        #             **metrics, # Include all calculated validation metrics
        #             'S_cache': S,
        #             # Add current experiment parameters for tracking
        #         }
        #         append_log(training_log_path, log_row)
            
        #     # Checkpoint save
        #     ckpt = {
        #         "epoch": epoch,
        #         "model_state": model.state_dict(),
        #         "optimizer_state": optimizer.state_dict(),
        #         "config": config.__dict__,
        #     }
        #     os.makedirs(config.checkpoint_dir, exist_ok=True)
        #     torch.save(ckpt, os.path.join(config.checkpoint_dir, f"ckpt_epoch_{epoch:03d}.pt"))
        # Optional validation + checkpoint
        if (val_dataset_q is not None) and (epoch % config.validate_every == 0):
            model.eval()
            
            print("  Validation:")

            # --- EFFICIENT FIX: Pre-compute the Plug-in Mean-Opt policy ONCE ---
            p_bar_global = p_pool.mean(axis=0)
            lambda_bar_global = float(lam_pool.mean())
            
            x_opt_mean_plugin = mean_opt_plug_in(
                p_bar=p_bar_global,
                lambda_bar=lambda_bar_global,
                S=S,
                compute_utility_fn=compute_utility_fn,
                project_fn=project_fn,
                M=M,
                device=device,
                verbose=False # Keep this off for cleaner logs
            )

            # --- Define all policies for evaluation ---
            policies_to_evaluate = {
                "RL2O-CVaR": model,
                
                # Now, the lambda just returns the pre-computed policy
                "Plug-in Mean-Opt": lambda q: x_opt_mean_plugin,

                "Popularity Heuristic (Top-S)": lambda q: popularity_deterministic_topS(
                    p_bar=q[:M], # Assumes q[:M] is the popularity estimate
                    S=int(round(S))
                )
            }
            
            # --- Run evaluation for all policies ---
            all_metrics = evaluate_all_policies(
                policies_to_evaluate=policies_to_evaluate,
                val_dataset_q=val_dataset_q,
                env_pool_list=list(zip(p_pool, lam_pool)),
                project_fn=project_fn,
                compute_utility_fn=compute_utility_fn,
                S=S,
                gamma=config.gamma_tail,
                device=torch.device(device),
                rng=rng,
                n_env_eval=1000  # Number of environment samples for evaluation
            )

            # --- Log and print results ---
            flattened_metrics = {}
            for policy_name, metrics in all_metrics.items():
                print(f"    {policy_name}:")
                for metric_name, value in metrics.items():
                    new_key = f"{metric_name}_{policy_name}"
                    flattened_metrics[new_key] = value
                    print(f"      {metric_name}: {value:.4f}")

            # For logging, we can log the metrics of the main model
            # main_model_metrics = all_metrics.get("RL2O-CVaR", {})
            log_row = {
                'epoch': epoch,
                'loss_train_mean': epoch_loss/max(1,num_batches),
                'gamma_tail': config.gamma_tail,
                'tau': config.tau,
                'S_cache': S,
            }
            # print(flattened_metrics)
            log_row.update(flattened_metrics)
            append_log(training_log_path, log_row)
            
            # Checkpoint save
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config.__dict__,
            }
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(config.checkpoint_dir, f"ckpt_epoch_{epoch:03d}.pt"))


    # final save
    torch.save({"epoch": config.epochs, "model_state": model.state_dict()},
               os.path.join(config.checkpoint_dir, "final_model.pt"))