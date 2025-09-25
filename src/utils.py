# src/utils.py
"""
Utility helpers for Digital-Twin-assisted CVaR-robust D2D caching experiments.

This module contains lightweight functions that are used across the codebase:
  - reproducibility: set_seed, get_rng
  - device selection: get_device
  - tensor/array conversions: to_numpy, to_torch
  - checkpoint save/load for PyTorch models
  - ensure_dir helper
  - simple CSV logger append
  - small RNG wrapper helpers for sampling indices
  - batching helper to iterate over dataset in minibatches
  - timing context manager
  - metric helpers: mean, var, cvar (small wrappers)
  - collating env_pool to arrays for vectorized evaluation

Design goals:
  - clear minimal API that students can read and understand easily
  - robust to CPU/GPU setups
  - no heavy external dependencies beyond numpy/torch/standard library
"""

from __future__ import annotations

import csv
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility and device helpers
# ---------------------------------------------------------------------------
_GLOBAL_RNG = None

def set_seed(seed: int) -> np.random.Generator:
    """
    Set the random seed for numpy, torch, and python's RNG.
    Returns: A new numpy.random.Generator instance for consistent use.
    """
    # Use a single, modern RNG instance
    global _GLOBAL_RNG
    _GLOBAL_RNG = np.random.default_rng(seed)
    
    # Set other seeds for consistency
    torch.manual_seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return _GLOBAL_RNG

def get_rng() -> np.random.Generator:
    """
    Return the global numpy.random.Generator instance.
    Raises: RuntimeError if seed has not been set.
    """
    global _GLOBAL_RNG
    if _GLOBAL_RNG is None:
        raise RuntimeError("RNG has not been seeded. Call set_seed() first.")
    return _GLOBAL_RNG

def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Utility to select a torch device.

    Args:
        prefer_cuda: if True and CUDA is available, return 'cuda', else 'cpu'

    Returns:
        torch.device instance
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Conversions & small helpers
# ---------------------------------------------------------------------------

def to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a NumPy array on CPU.

    Args:
        t: torch.Tensor

    Returns:
        numpy.ndarray
    """
    if not torch.is_tensor(t):
        raise TypeError("to_numpy expects a torch.Tensor")
    return t.detach().cpu().numpy()


def to_torch(a: np.ndarray, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert a NumPy array (or anything convertible) to a torch tensor.

    Args:
        a: array-like
        device: torch.device (if None, uses CPU)
        dtype: torch.dtype

    Returns:
        torch.Tensor
    """
    if torch.is_tensor(a):
        return a
    t = torch.as_tensor(np.asarray(a), dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


# ---------------------------------------------------------------------------
# File system helpers (checkpoints, dirs, logging)
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists (create if necessary).

    Args:
        path: directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extras: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a PyTorch checkpoint containing model state and optional optimizer state.

    Args:
        path: file path (e.g., 'checkpoints/model_latest.pt')
        model: torch.nn.Module
        optimizer: optimizer (optional)
        epoch: epoch index or training step (optional)
        extras: other metadata to store
    """
    ensure_dir(os.path.dirname(path) or ".")
    state = {
        "model_state": model.state_dict(),
        "meta": {"saved_at": time.time(), "epoch": epoch}
    }
    if optimizer is not None:
        state["optim_state"] = optimizer.state_dict()
    if extras is not None:
        state["extras"] = extras
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load a checkpoint into model and (optionally) optimizer.

    Args:
        path: checkpoint file path
        model: torch.nn.Module instance to load weights into
        optimizer: optional optimizer to restore state
        device: optional torch.device to map checkpoint to (defaults to CPU)

    Returns:
        dict: metadata and extra fields saved in the checkpoint
    """
    if device is None:
        device = torch.device("cpu")
    chk = torch.load(path, map_location=device)
    model.load_state_dict(chk["model_state"])
    if optimizer is not None and "optim_state" in chk:
        optimizer.load_state_dict(chk["optim_state"])
    return chk.get("extras", {"meta": chk.get("meta", {})})


def append_row_csv(path: str, row: Dict[str, Any], header: Optional[List[str]] = None) -> None:
    """
    Append a row (dict) to a CSV file. If the file does not exist, create it and write header.

    Args:
        path: csv file path
        row: dict mapping column name -> value
        header: optional explicit header order (list of column names). If None, use row.keys()
    """
    ensure_dir(os.path.dirname(path) or ".")
    file_exists = os.path.exists(path)
    keys = header if header is not None else list(row.keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in keys})


# ---------------------------------------------------------------------------
# RNG / sampling helpers
# ---------------------------------------------------------------------------

def rng_generator(seed: Optional[int] = None) -> np.random.Generator:
    """
    Get a numpy.random.Generator with optional seed.

    Args:
        seed: optional integer seed

    Returns:
        numpy.random.Generator
    """
    return np.random.default_rng(seed)


def sample_indices(pool_size: int, n: int, rng: Optional[np.random.Generator] = None, replace: bool = True) -> np.ndarray:
    """
    Sample indices from 0..pool_size-1 reproducibly.

    Args:
        pool_size: size of the pool
        n: number of indices to sample
        rng: optional numpy.random.Generator
        replace: whether to sample with replacement

    Returns:
        numpy array of indices (dtype=int)
    """
    if rng is None:
        rng = rng_generator(None)
    return rng.integers(0, pool_size, size=n) if replace else rng.choice(pool_size, size=n, replace=False)


# ---------------------------------------------------------------------------
# Dataset / batching helpers
# ---------------------------------------------------------------------------

def minibatch_indices(n_total: int, batch_size: int, shuffle: bool = True, rng: Optional[np.random.Generator] = None) -> Iterator[np.ndarray]:
    """
    Yield arrays of indices for minibatching.

    Args:
        n_total: total number of samples
        batch_size: desired batch size
        shuffle: whether to shuffle order
        rng: optional RNG for shuffle reproducibility

    Yields:
        numpy arrays of indices for each minibatch
    """
    if rng is None:
        rng = rng_generator(None)
    idx = np.arange(n_total)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n_total, batch_size):
        yield idx[start: start + batch_size]


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def timing(name: str = "block") -> Iterator[float]:
    """
    Context manager for timing code blocks.

    Usage:
        with timing("train-epoch") as t:
            ... do work ...
        print("elapsed:", t)

    On exit yields the elapsed seconds (float).
    """
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        print(f"[timing] {name}: {elapsed:.3f} s")


# ---------------------------------------------------------------------------
# Small metric helpers (convenience wrappers)
# ---------------------------------------------------------------------------

def mean_metric(u: Sequence[float]) -> float:
    """Return mean of sequence (returns nan for empty)."""
    u = np.asarray(u)
    if u.size == 0:
        return float("nan")
    return float(u.mean())


def var_metric(u: Sequence[float], gamma: float) -> float:
    """Return empirical VaR (gamma-quantile, lower tail)."""
    u = np.asarray(u)
    if u.size == 0:
        return float("nan")
    return float(np.quantile(u, gamma, interpolation="linear"))


def cvar_metric(u: Sequence[float], gamma: float) -> float:
    """Return empirical CVaR (average of worst gamma fraction)."""
    u = np.asarray(u)
    if u.size == 0:
        return float("nan")
    L = u.size
    k = max(1, int(np.ceil(gamma * L)))
    return float(np.mean(np.sort(u)[:k]))


# ---------------------------------------------------------------------------
# Environment pool collate (vectorization helper)
# ---------------------------------------------------------------------------

def collate_env_pool(env_pool: Sequence[Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert env_pool (sequence of (p_vector, lambda_scalar)) into two arrays:
      - P: shape (N, M)
      - LAM: shape (N,)

    This is helpful for vectorized evaluation across many posterior draws.

    Args:
        env_pool: sequence of (p: np.ndarray (M,), lam: float)

    Returns:
        (P_array, lambda_array)
    """
    if len(env_pool) == 0:
        return np.empty((0, 0)), np.empty((0,))
    P_list = []
    lam_list = []
    for p, lam in env_pool:
        P_list.append(np.asarray(p))
        lam_list.append(float(lam))
    P_arr = np.stack(P_list, axis=0)
    lam_arr = np.asarray(lam_list, dtype=float)
    return P_arr, lam_arr


# ---------------------------------------------------------------------------
# Small I/O helpers (json)
# ---------------------------------------------------------------------------

def save_json(path: str, obj: Any, indent: int = 2) -> None:
    """Save obj as JSON to path (create directories if needed)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str) -> Any:
    """Load JSON object from file."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Example usage in a training/evaluation loop
# ---------------------------------------------------------------------------
# from src.utils import set_seed, get_device, timing, minibatch_indices, save_checkpoint
#
# set_seed(42)
# dev = get_device()
# with timing("toy-run"):
#     for idx in minibatch_indices(len(dataset), batch_size=32):
#         ...
#
# save_checkpoint("checkpoints/model.pt", model, optimizer, epoch=10)
# ---------------------------------------------------------------------------
