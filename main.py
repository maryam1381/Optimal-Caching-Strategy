# #!/usr/bin/env python3
"""
main.py — Top-level experiment orchestration for the Digital-Twin-assisted
CVaR-robust D2D caching project.

Usage (from repository root):
    python main.py --help

Typical example:
    # run with defaults and small toy settings
    python main.py \
        --seed 42 \
        --save-dir experiments/checkpoints/run1 \
        --epochs 50 \
        --N-pool 2000 \
        --dataset-size 1000

What this script does:
  1. Parse CLI args (and optional JSON config).
  2. Set seeds and device.
  3. Build or load an environment posterior pool (p vectors and lambda samples).
  4. Construct a simple synthetic measurement dataset (q features) from the pool:
       - q contains observed request counts (or normalized frequencies) and a count K
  5. Build the MLP model (from src/model.py).
  6. Call the training routine (train.train) with the prepared components.
  7. Save config and sample outputs.

IMPORTANT:
- This script assumes the following functions exist in the `src` package:
    - src.utils: set_seed, get_device, ensure_dir, save_json
    - src.env_pool: build_env_pool
    - src.model: MLP
    - src.train: train  (entrypoint to run training)
  If your function names differ, adapt the imports / call below.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import inspect
import numpy as np
import torch

from src.losses import project_capped_simplex, utility_from_env_samples
from src.train import TrainConfig, evaluate_model_simple

# Make repo root and src discoverable even if user runs script from another cwd
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(REPO_ROOT))

# Try to import local modules; if missing, provide friendly messages / minimal fallbacks
missing = []
try:
    from src import env_pool as env_pool_mod
except Exception as e:
    env_pool_mod = None
    missing.append(("src.env_pool", e))

try:
    from src import model as model_mod
except Exception as e:
    model_mod = None
    missing.append(("src.model", e))

try:
    from src import train as train_mod
except Exception as e:
    train_mod = None
    missing.append(("src.train", e))

try:
    from src import utils as utils_mod
except Exception as e:
    utils_mod = None
    missing.append(("src.utils", e))


# Minimal fallback utilities if src.utils is missing (so error messages are clearer)
if utils_mod is None:
    class _UtilsFallback:
        @staticmethod
        def set_seed(seed: int):
            np.random.seed(int(seed))
            import random
            random.seed(int(seed))
            try:
                import torch
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            except Exception:
                pass

        @staticmethod
        def get_device(prefer_cuda: bool = True):
            if prefer_cuda and torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        @staticmethod
        def ensure_dir(path: str):
            Path(path).mkdir(parents=True, exist_ok=True)

        @staticmethod
        def save_json(path: str, obj: Dict[str, Any]):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)

    utils_mod = _UtilsFallback()
    print("[main] WARNING: src.utils not found — using fallback minimal utils. "
          "Prefer implementing src/utils.py for full behavior.")


# If any critical modules are missing, provide a clear error before long runtime
if env_pool_mod is None or model_mod is None or train_mod is None:
    msg_lines = ["One or more required src modules could not be imported:"]
    if env_pool_mod is None:
        msg_lines.append(" - src.env_pool (needed to build environment pool)")
    if model_mod is None:
        msg_lines.append(" - src.model (needed to create the MLP model)")
    if train_mod is None:
        msg_lines.append(" - src.train (needed to run training)")
    msg_lines.append("")
    msg_lines.append("Make sure you run this script from the repository root and that 'src/' exists.")
    msg_lines.append(f"Repo root: {REPO_ROOT}")
    raise ImportError("\n".join(msg_lines))


# -------------------------
# Default experiment config
# -------------------------
DEFAULT_CONFIG = {
    "M": 100,
    "W": 200,
    "gamma_r": 0.8,
    "a0_p": 1.0,
    "a0_lambda": 1.0, "b0": 1.0,
    "A_user": 10.0,
    "lambda_true": 2.5,
    "N_pool": 2000,
    "dataset_size": 1000,
    "batch_size": 16,
    "L": 200,
    "gamma": 0.05,
    "tau": 1e-2,
    "lr": 1e-3,
    "epochs": 40,
    "S_cache": 10,
    "hidden_small": True,
    "dropout": 0.0,
    "seed": 42,
    "save_dir": "experiments/checkpoints/run_default",
}

# -------------------------
# Utility: build synthetic measurement dataset q
# -------------------------
def build_synthetic_dataset_from_pool(pool: list, dataset_size: int, M: int, W: int, A_user: float,
                                      lambda_true: float, rng: np.random.Generator):
    P_pool, LAM_pool = pool
    dataset = np.zeros((dataset_size, M + 1), dtype=np.float32)
    for i in range(dataset_size):
        idx = int(rng.integers(0, len(pool)))
        p_true = P_pool[idx]
        p_true = np.asarray(p_true, dtype=np.float64)
        n = rng.multinomial(W, p_true)
        K = int(rng.poisson(lam=lambda_true * A_user))
        n_norm = n / (W + 1e-12)
        K_norm = np.array([K / max(1.0, A_user)], dtype=np.float32)
        dataset[i, :M] = n_norm.astype(np.float32)
        dataset[i, M] = K_norm
    return dataset


def main(config: Dict[str, Any]):
    utils_mod.set_seed(int(config["seed"]))
    device = utils_mod.get_device(prefer_cuda=True)
    print(f"[main] using device: {device}; seed: {config['seed']}")

    save_dir = Path(config["save_dir"])
    utils_mod.ensure_dir(str(save_dir))

    # store the config for reproducibility
    cfg_path = save_dir / "config.json"
    utils_mod.save_json(str(cfg_path), config)
    print(f"[main] config saved to {cfg_path}")

    # -----------------------------
    # 1) Build posterior env pool (digital twin draws)
    # -----------------------------
    print("[main] building environment posterior pool...")

    pool_iter = env_pool_mod.build_env_pool_simulated(
        N_pool=int(config["N_pool"]),
        M=int(config["M"]),
        W=int(config["W"]),
        zipf_exponent=float(config["gamma_r"]),
        a0_p=float(config.get("a0_p", 1.0)),
        a0_lambda=float(config.get("a0_lambda", 1.0)),
        b0_lambda=float(config.get("b0", 1.0)),
        A_obs=float(config.get("A_user", 1.0)),
        lambda_true=float(config.get("lambda_true", 1.0)),
        rng_seed=int(config.get("seed", 0)),
        as_torch=False   # return NumPy / Python objects
    )
    # convert to list for repeated indexing
    env_pool = list(pool_iter)
    print(f"[main] env pool built: {len(env_pool)} samples")

    sample_save_path = save_dir / "env_pool_preview.npz"
    
    P_pool, LAM_pool = env_pool
    P_preview = np.stack([P_pool[:min(200, len(P_pool))]], axis=0)
    L_preview = np.array([LAM_pool[:min(200, len(LAM_pool))]])

    np.savez_compressed(str(sample_save_path), P=P_preview, L=L_preview)
    print(f"[main] saved env pool preview to {sample_save_path}")

    # -----------------------------
    # 2) Build measurement dataset (simple synthetic)
    # -----------------------------
    rng = np.random.default_rng(int(config["seed"]) + 1)
    dataset = build_synthetic_dataset_from_pool(
        pool=env_pool,
        dataset_size=int(config["dataset_size"]),
        M=int(config["M"]),
        W=int(config["W"]),
        A_user=float(config["A_user"]),
        lambda_true=float(config["lambda_true"]),
        rng=rng,
    )
    print(f"[main] synthetic dataset built: {dataset.shape}")


    # -----------------------------
    # 3) Build model (MLP)
    # -----------------------------
    print("[main] building model...")
    # Support model_mod.MLP returning model or (model, optimizer)
    model ,optimizer = model_mod.create_mlp(M=int(config["M"]),
                                                use_layernorm=True,
                                                dropout=float(config["dropout"]),
                                                lr=float(config["lr"]), 
                                                weight_decay=1e-4)

    model = model.to(device)
    print("[main] model created and moved to device.")

    # -----------------------------
    # 4) Call the training routine
    # -----------------------------
    pool_args = {
        "N_pool": int(config["N_pool"]),
        "M": int(config["M"]),
        "W": int(config["W"]),
        "gamma_r": float(config["gamma_r"]),
        "a0_p": float(config["a0_p"]),
        "a0_lambda": float(config["a0_lambda"]),
        "b0": float(config["b0"]),
        "A_user": float(config["A_user"]),
        "lambda_": float(config["lambda_true"]),
        "rng_seed": int(config["seed"]),
    }
    pout_args = {}

    print("[main] calling training routine...")
    # Try to call train with the signature used in the repo; fallback if signature differs
    try:
        # attempt the full signature first (preferred)
        train_config = TrainConfig(N_pool=int(config["N_pool"]),
                                    L=int(config["L"]),
                                    batch_size=int(config["batch_size"]),
                                    gamma_tail=float(config["gamma"]),
                                    tau=float(config["tau"]),
                                    epochs=int(config["epochs"]),
                                    lr=float(config["lr"]),
                                    device="cpu",
                                    checkpoint_dir="./checkpoints_toy", 
                                    verbose=True)

        train_mod.train_loop(
            model=model,
            optimizer= optimizer,
            env_pool=env_pool,
            dataset_q=dataset,
            S=int(config["S_cache"]),
            compute_utility_fn=utility_from_env_samples,
            project_fn=project_capped_simplex,
            config=train_config,
            val_dataset_q=dataset[:20],
            eval_fn=evaluate_model_simple
        )
    except TypeError:
        print(f"[Error] in train loop at the 301.")

    print("[main] training finished. Check the save directory for outputs.")

# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train CVaR-robust D2D caching policy (digital twin).")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file to override defaults.")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_CONFIG["save_dir"], help="Directory to save checkpoints and config.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="Training epochs.")
    parser.add_argument("--N-pool", type=int, dest="N_pool", default=DEFAULT_CONFIG["N_pool"], help="Posterior pool size.")
    parser.add_argument("--dataset-size", type=int, default=DEFAULT_CONFIG["dataset_size"], help="Number of synthetic measurements.")
    parser.add_argument("--M", type=int, default=DEFAULT_CONFIG["M"], help="Number of files.")
    parser.add_argument("--W", type=int, default=DEFAULT_CONFIG["W"], help="Measurement window (requests).")
    parser.add_argument("--L", type=int, default=DEFAULT_CONFIG["L"], help="Posterior draws per measurement in training.")
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=DEFAULT_CONFIG["batch_size"], help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"], help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["gamma"], help="CVaR tail level.")
    parser.add_argument("--S-cache", type=int, dest="S_cache", default=DEFAULT_CONFIG["S_cache"], help="Cache size S.")
    parser.add_argument("--lambda-true", type=float, dest="lambda_true", default=DEFAULT_CONFIG["lambda_true"], help="Nominal true user density for dataset generation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # load JSON config if provided
    cfg = DEFAULT_CONFIG.copy()
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        cfg.update(cfg_json)
    cfg.update({
        "save_dir": args.save_dir,
        "seed": args.seed,
        "epochs": args.epochs,
        "N_pool": args.N_pool,
        "dataset_size": args.dataset_size,
        "M": args.M,
        "W": args.W,
        "L": args.L,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "gamma": args.gamma,
        "S_cache": args.S_cache,
        "lambda_true": args.lambda_true,
    })
    main(cfg)
