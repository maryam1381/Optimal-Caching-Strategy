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
from src.utils import to_torch, append_row_csv as append_log

# Project utils assumed in losses.py
from src.losses import single_utility_from_env, smoothed_cvar_loss_from_utilities, project_capped_simplex, utility_from_env_samples
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
    eval_fn: Optional[Callable] = None, # <-- UPDATED: eval_fn is now used
) -> None:
    """
    Main training loop.

    Args:
      model: PyTorch model mapping q -> raw scores y (shape (batch, M))
      optimizer: optimizer for model parameters
      env_pool: tuple of numpy arrays (P_pool, LAM_pool)
      dataset_q: numpy array of measurements q (shape [N_measurements, input_dim]) (TRAIN SET)
      S: cache capacity (expected). Projection target sum <= S.
      compute_utility_fn: callable (p:torch.Tensor, lam:torch.Tensor, x:torch.Tensor) -> utility scalar
                          (This is the BATCHED version for training)
      : callable project_capped_simplex(y: torch.Tensor, S: float) -> x:torch.Tensor
      config: TrainConfig dataclass
      rng: numpy.random.Generator instance for reproducibility
      val_dataset_q: optional validation dataset (numpy)
      eval_fn: optional evaluation function (e.g., evaluate_model) used for validation
    """
    device = torch.device(config.device)
    model.to(device)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    N = dataset_q.shape[0]
    idx_all = np.arange(N)

    # --- stack env pool to arrays for fast gather ---
    p_pool, lam_pool = env_pool
    N_pool = len(p_pool)
    assert N_pool > 0, "env_pool is empty"
    
    # Create the list version of the env_pool once for validation
    # This avoids rebuilding it every validation epoch
    env_pool_list_val = None
    if val_dataset_q is not None and eval_fn is not None:
         env_pool_list_val = list(zip(p_pool, lam_pool))

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
            q_batch = dataset_q[sel]  # (B, in_dim)
            B = q_batch.shape[0]

            optimizer.zero_grad()

            # (1) Forward entire batch → y → project to x
            q_t = torch.from_numpy(q_batch).float().to(device)  # (B, in_dim)
            y = model(q_t,)

            x_rows = [project_fn(y[b], S) for b in range(B)]

            x = torch.stack(x_rows, dim=0)  # (B, M)

            # (2) Sample L envs per batch row, gather p, lambda
            inds = sample_env_indices(B, config.L)  # (B, L)
            p_batch_np = p_pool[inds]  # (B, L, M)
            lam_batch_np = lam_pool[inds]  # (B, L)

            p_batch = to_torch(p_batch_np, device)  # (B, L, M)
            lam_batch = to_torch(lam_batch_np, device)  # (B, L)

            # (3) Compute utilities for all (B, L) scenarios (vectorized)
            U = compute_utility_fn(
                p_samples=p_batch,  # (B, L, M)
                lam_samples=lam_batch,  # (B, L)
                x=x  # (B, M)
            )

            # (4) Compute CVaR loss from utilities
            loss, _ = smoothed_cvar_loss_from_utilities(U, gamma=config.gamma_tail, tau=config.tau)

            # (5) Backprop, clip, step
            loss.backward(retain_graph=True)
            # for n, p in model.named_parameters():
            #     print(n, p.grad.abs().mean().item())

            if config.clip_grad_norm and config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            num_batches += 1

            # Removed metric logging from here, it was logging last-batch training metrics

        # --- VALIDATION BLOCK (UPDATED) ---
        if (epoch % config.validate_every == 0):
            log_row = {
                'epoch': epoch,
                'loss_train_mean': epoch_loss / max(1, num_batches),
                'gamma_tail': config.gamma_tail,
                'tau': config.tau,
                'S_cache': S,
            }
            
            # Run validation if assets are provided
            if (val_dataset_q is not None) and (eval_fn is not None) and (len(val_dataset_q) > 0):
                model.eval() # Set model to evaluation mode
                
                # Run the external evaluation function
                # Note: eval_fn (evaluate_model) uses n_env_eval=500 by default
                val_metrics = eval_fn(
                    model=model,
                    val_dataset_q=val_dataset_q,
                    env_pool_list=env_pool_list_val,
                    project_fn=project_fn,
                    compute_utility_fn=single_utility_from_env, # <-- Use the single-sample utility fn
                    S=S,
                    gamma=config.gamma_tail, # This is the loss-tail gamma (1-alpha)
                    device=device,
                    rng=rng
                )
                
                # Add validation metrics to log row
                log_row['val_mean_utility'] = val_metrics.get('mean_utility_mean', np.nan)
                log_row['val_mean_loss'] = val_metrics.get('mean_loss_mean', np.nan)
                # Note: evaluate_model returns metrics named after gamma (e.g., CVaR_0.05)
                log_row['val_CVaR_loss'] = val_metrics.get(f'CVaR_{config.gamma_tail}', np.nan)
                log_row['val_VaR_loss'] = val_metrics.get(f'VaR_{config.gamma_tail}', np.nan)
                
                # Checkpoint save (only save if we ran validation)
                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config.__dict__,
                }
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                torch.save(ckpt, os.path.join(config.checkpoint_dir, f"ckpt_epoch_{epoch:03d}.pt"))

            # Append the log row (either with or without validation metrics)
            append_log(training_log_path, log_row)

            if config.verbose:
                val_loss_str = f"val_loss={log_row.get('val_mean_loss', 'N/A'):.6f}" if 'val_mean_loss' in log_row else ""
                print(f"[Epoch {epoch:03d}] loss={log_row['loss_train_mean']:.6f} {val_loss_str} "
                      f"time={time.time() - t0:.1f}s")
        
        # --- END VALIDATION BLOCK ---

    # Final save
    torch.save({"epoch": config.epochs, "model_state": model.state_dict()},
               os.path.join(config.checkpoint_dir, "final_model.pt"))