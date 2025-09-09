# #!/usr/bin/env python3
# """
# main.py — Top-level experiment orchestration for the Digital-Twin-assisted
# CVaR-robust D2D caching project.

# Usage (from repository root):
#     python main.py --help

# Typical example:
#     # run with defaults and small toy settings
#     python main.py \
#         --seed 42 \
#         --save-dir experiments/checkpoints/run1 \
#         --epochs 50 \
#         --N-pool 2000 \
#         --dataset-size 1000

# What this script does:
#   1. Parse CLI args (and optional JSON config).
#   2. Set seeds and device.
#   3. Build or load an environment posterior pool (p vectors and lambda samples).
#   4. Construct a simple synthetic measurement dataset (q features) from the pool:
#        - q contains observed request counts (or normalized frequencies) and a count K
#   5. Build the MLP model (from src/model.py).
#   6. Call the training routine (train.train) with the prepared components.
#   7. Save config and sample outputs.

# IMPORTANT:
# - This script assumes the following functions exist in the `src` package:
#     - src.utils: set_seed, get_device, ensure_dir, save_json
#     - src.env_pool: build_env_pool
#     - src.model: MLP
#     - src.train: train  (entrypoint to run training)
#   If your function names differ, adapt the imports / call below.
# """

# import argparse
# import json
# import os
# import time
# from pathlib import Path
# from typing import Dict, Any

# import numpy as np
# import torch

# # local modules (from src/)
# from src import env_pool as env_pool_mod
# from src import model as model_mod
# from src import train as train_mod
# from src import utils as utils_mod


# # -------------------------
# # Default experiment config
# # -------------------------
# DEFAULT_CONFIG = {
#     # data / pool settings
#     "M": 100,                # number of files
#     "W": 200,                # measurement window (number of requests observed)
#     "gamma_r": 0.8,          # Zipf param used to generate synthetic true environments (only for env pool)
#     "a0": 1.0, "b0": 1.0,    # Gamma prior hyperparameters for lambda
#     "A_user": 10.0,          # monitored area (for user sightings)
#     "lambda_true": 2.5,      # nominal "true" user density used to sample K when generating dataset
#     "N_pool": 2000,          # number of posterior scenario draws in env pool
#     # training settings
#     "dataset_size": 1000,    # number of measurements q in dataset
#     "batch_size": 16,
#     "L": 200,                # number of posterior draws per measurement used at training time
#     "gamma": 0.05,           # CVaR tail level
#     "tau": 1e-2,             # softplus smoothing parameter
#     "lr": 1e-3,
#     "epochs": 100,
#     "S_cache": 10,           # cache capacity (files)
#     # model
#     "hidden_small": True,
#     "dropout": 0.0,
#     # bookkeeping
#     "seed": 42,
#     "save_dir": "experiments/checkpoints/run_default",
# }


# # -------------------------
# # Utility: build synthetic measurement dataset q
# # -------------------------
# def build_synthetic_dataset_from_pool(pool: list, dataset_size: int, M: int, W: int, A_user: float, lambda_true: float, rng: np.random.Generator):
#     """
#     Build a simple measurement dataset: each measurement q is a vector of length (M+1)
#     formed by (normalized observed request counts n / W, normalized observed user count K / A_user).
#     The digital twin expects counts (n, K) when constructing posteriors; here we keep
#     features normalized to [0,1] for the neural network input.

#     Args:
#         pool: env_pool list (each element is (p_vector, lambda_scalar)) — used as a source of 'true' p
#         dataset_size: number of measurement samples to generate
#         M, W: measurement parameters (files and requests)
#         A_user, lambda_true: used to sample observed K (Poisson)
#         rng: numpy RNG

#     Returns:
#         dataset: numpy array of shape (dataset_size, M+1)
#     """
#     dataset = np.zeros((dataset_size, M + 1), dtype=np.float32)

#     for i in range(dataset_size):
#         # pick a random ground-truth popularity vector from the pool as the 'truth'
#         idx = rng.integers(0, len(pool))
#         p_true, lam_true_sample = pool[idx]
#         p_true = np.asarray(p_true, dtype=np.float64)

#         # sample W requests from the true popularity (multinomial)
#         n = rng.multinomial(W, p_true)

#         # sample K (user sightings) from Poisson(lambda_true * A_user)
#         K = rng.poisson(lam=lambda_true * A_user)

#         # normalize
#         n_norm = n / (W + 1e-12)        # empirical frequencies
#         K_norm = np.array([K / max(1.0, A_user)], dtype=np.float32)

#         dataset[i, :M] = n_norm.astype(np.float32)
#         dataset[i, M] = K_norm  # last element is normalized K

#     return dataset


# # -------------------------
# # Main
# # -------------------------
# def main(config: Dict[str, Any]):
#     utils_mod.set_seed(int(config["seed"]))
#     device = utils_mod.get_device(prefer_cuda=True)
#     print(f"[main] using device: {device}; seed: {config['seed']}")

#     save_dir = Path(config["save_dir"])
#     utils_mod.ensure_dir(str(save_dir))

#     # store the config for reproducibility
#     cfg_path = save_dir / "config.json"
#     utils_mod.save_json(str(cfg_path), config)
#     print(f"[main] config saved to {cfg_path}")

#     # -----------------------------
#     # 1) Build posterior env pool (digital twin draws)
#     # -----------------------------
#     print("[main] building environment posterior pool...")
#     # env_pool_mod.build_env_pool should return an iterable of (p_vector, lambda_scalar) pairs
#     pool_iter = env_pool_mod.build_env_pool(
#         N_pool=int(config["N_pool"]),
#         M=int(config["M"]),
#         W=int(config["W"]),
#         gamma_r=float(config["gamma_r"]),
#         a0=float(config["a0"]),
#         b0=float(config["b0"]),
#         A_user=float(config["A_user"]),
#         lambda_=float(config["lambda_true"]),
#         rng_seed=int(config["seed"]),
#     )
#     # convert to list for repeated indexing
#     env_pool = list(pool_iter)
#     print(f"[main] env pool built: {len(env_pool)} samples")

#     # Optionally save a small portion of the pool for inspection (first 50 samples)
#     sample_save_path = save_dir / "env_pool_preview.npz"
#     P_preview = np.stack([p for p, _ in env_pool[:min(200, len(env_pool))]], axis=0)
#     L_preview = np.array([lam for _, lam in env_pool[:min(200, len(env_pool))]])
#     np.savez_compressed(str(sample_save_path), P=P_preview, L=L_preview)
#     print(f"[main] saved env pool preview to {sample_save_path}")

#     # -----------------------------
#     # 2) Build measurement dataset (simple synthetic)
#     # -----------------------------
#     rng = np.random.default_rng(int(config["seed"]) + 1)
#     dataset = build_synthetic_dataset_from_pool(
#         pool=env_pool,
#         dataset_size=int(config["dataset_size"]),
#         M=int(config["M"]),
#         W=int(config["W"]),
#         A_user=float(config["A_user"]),
#         lambda_true=float(config["lambda_true"]),
#         rng=rng,
#     )
#     print(f"[main] synthetic dataset built: {dataset.shape}")

#     # -----------------------------
#     # 3) Build model (MLP)
#     # -----------------------------
#     print("[main] building model...")
#     # Our model.MLP factory returns (model, optimizer) in earlier helper script; adjust if yours differs.
#     model, optimizer = model_mod.MLP(
#         M=int(config["M"]),
#         layernorm=True,
#         dropout=float(config["dropout"]),
#         lr=float(config["lr"]),
#         weight_decay=1e-4
#     )
#     model = model.to(device)
#     print("[main] model created and moved to device.")

#     # -----------------------------
#     # 4) Call the training routine
#     # -----------------------------
#     # NOTE: adapt the following call to match the exact signature in your src/train.py
#     print("[main] calling training routine...")

#     # prepare pool_args to pass into train (train implementation expects these)
#     pool_args = {
#         "N_pool": int(config["N_pool"]),
#         "M": int(config["M"]),
#         "W": int(config["W"]),
#         "gamma_r": float(config["gamma_r"]),
#         "a0": float(config["a0"]),
#         "b0": float(config["b0"]),
#         "A_user": float(config["A_user"]),
#         "lambda_": float(config["lambda_true"]),
#         "rng_seed": int(config["seed"]),
#     }

#     # pout_args (optional): any arguments your pout or utility functions expect
#     pout_args = {}

#     # For the train.train signature that we used previously, it expected:
#     # train(model, dataset, N_pool, L, B, gamma, tau, lr, epochs, S, pool_args, pout_args, val_dataset)
#     try:
#         train_mod.train(
#             model=model,
#             dataset=dataset,
#             N_pool=int(config["N_pool"]),
#             L=int(config["L"]),
#             B=int(config["batch_size"]),
#             gamma=float(config["gamma"]),
#             tau=float(config["tau"]),
#             lr=float(config["lr"]),
#             epochs=int(config["epochs"]),
#             S=int(config["S_cache"]),
#             pool_args=pool_args,
#             pout_args=pout_args,
#             val_dataset=None,    # optionally provide a validation split
#             env_pool=env_pool    # if your train function accepts a pre-built pool, pass it
#         )
#     except TypeError:
#         # If train has a different signature, attempt a simpler call (student should adapt)
#         print("[main] train.train signature mismatch — attempting fallback call.")
#         train_mod.train(model, dataset, pool_args, save_dir)

#     print("[main] training finished. Check the save directory for outputs.")


# # -------------------------
# # CLI
# # -------------------------
# def parse_args():
#     parser = argparse.ArgumentParser(description="Train CVaR-robust D2D caching policy (digital twin).")
#     parser.add_argument("--config", type=str, default=None, help="Optional JSON config file to override defaults.")
#     parser.add_argument("--save-dir", type=str, default=DEFAULT_CONFIG["save_dir"], help="Directory to save checkpoints and config.")
#     parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed.")
#     parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="Training epochs.")
#     parser.add_argument("--N-pool", type=int, dest="N_pool", default=DEFAULT_CONFIG["N_pool"], help="Posterior pool size.")
#     parser.add_argument("--dataset-size", type=int, default=DEFAULT_CONFIG["dataset_size"], help="Number of synthetic measurements.")
#     parser.add_argument("--M", type=int, default=DEFAULT_CONFIG["M"], help="Number of files.")
#     parser.add_argument("--W", type=int, default=DEFAULT_CONFIG["W"], help="Measurement window (requests).")
#     parser.add_argument("--L", type=int, default=DEFAULT_CONFIG["L"], help="Posterior draws per measurement in training.")
#     parser.add_argument("--batch-size", type=int, dest="batch_size", default=DEFAULT_CONFIG["batch_size"], help="Mini-batch size.")
#     parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"], help="Learning rate.")
#     parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["gamma"], help="CVaR tail level.")
#     parser.add_argument("--S-cache", type=int, dest="S_cache", default=DEFAULT_CONFIG["S_cache"], help="Cache size S.")
#     parser.add_argument("--lambda-true", type=float, dest="lambda_true", default=DEFAULT_CONFIG["lambda_true"], help="Nominal true user density for dataset generation.")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()

#     # load JSON config if provided
#     cfg = DEFAULT_CONFIG.copy()
#     if args.config is not None:
#         with open(args.config, "r") as f:
#             cfg_json = json.load(f)
#         cfg.update(cfg_json)
#     # override with CLI flags
#     cfg.update({
#         "save_dir": args.save_dir,
#         "seed": args.seed,
#         "epochs": args.epochs,
#         "N_pool": args.N_pool,
#         "dataset_size": args.dataset_size,
#         "M": args.M,
#         "W": args.W,
#         "L": args.L,
#         "batch_size": args.batch_size,
#         "lr": args.lr,
#         "gamma": args.gamma,
#         "S_cache": args.S_cache,
#         "lambda_true": args.lambda_true,
#     })

#     main(cfg)






























#!/usr/bin/env python3
"""
main.py — Top-level experiment orchestration (robust and Windows-friendly).

Run from repository root:
    python main.py --help
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
    "a0": 1.0, "b0": 1.0,
    "A_user": 10.0,
    "lambda_true": 2.5,
    "N_pool": 2000,
    "dataset_size": 1000,
    "batch_size": 16,
    "L": 200,
    "gamma": 0.05,
    "tau": 1e-2,
    "lr": 1e-3,
    "epochs": 100,
    "S_cache": 10,
    "hidden_small": True,
    "dropout": 0.0,
    "seed": 42,
    "save_dir": "experiments/checkpoints/run_default",
}


def build_synthetic_dataset_from_pool(pool: list, dataset_size: int, M: int, W: int, A_user: float,
                                      lambda_true: float, rng: np.random.Generator):
    dataset = np.zeros((dataset_size, M + 1), dtype=np.float32)
    for i in range(dataset_size):
        idx = int(rng.integers(0, len(pool)))
        p_true, lam_true_sample = pool[idx]
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

    cfg_path = save_dir / "config.json"
    utils_mod.save_json(str(cfg_path), config)
    print(f"[main] config saved to {cfg_path}")

    print("[main] building environment posterior pool...")
    # pool_iter = env_pool_mod.build_env_pool_simulated(
    #     N_pool=int(config["N_pool"]),
    #     M=int(config["M"]),
    #     W=int(config["W"]),
    #     zipf_exponent=float(config["gamma_r"]),
    #     a0=float(config["a0"]),
    #     b0=float(config["b0"]),
    #     A_user=float(config["A_user"]),
    #     lambda_=float(config["lambda_true"]),
    #     rng_seed=int(config["seed"]),
    # )
    pool_iter = env_pool_mod.build_env_pool_simulated(
        N_pool=int(config["N_pool"]),
        M=int(config["M"]),
        W=int(config["W"]),
        zipf_exponent=float(config["gamma_r"]),
        a0_p=float(config.get("a0", 1.0)),
        a0_lambda=float(config.get("a0", 1.0)),
        b0_lambda=float(config.get("b0", 1.0)),
        A_obs=float(config.get("A_user", 1.0)),
        lambda_true=float(config.get("lambda_true", 1.0)),
        rng_seed=int(config.get("seed", 0)),
        as_torch=False   # return NumPy / Python objects
    )
    env_pool = list(pool_iter)

    env_pool = list(pool_iter)
    print(f"[main] env pool built: {len(env_pool)} samples")

    sample_save_path = save_dir / "env_pool_preview.npz"
    P_preview = np.stack([p for p, _ in env_pool[:min(200, len(env_pool))]], axis=0)
    L_preview = np.array([lam for _, lam in env_pool[:min(200, len(env_pool))]])
    np.savez_compressed(str(sample_save_path), P=P_preview, L=L_preview)
    print(f"[main] saved env pool preview to {sample_save_path}")

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

    print("[main] building model...")
    # Support model_mod.MLP returning model or (model, optimizer)
    mlp_args = dict(
        M=int(config["M"]),
        layernorm=True,
        dropout=float(config["dropout"])
    )
    # If model factory expects lr or weight_decay, it's ok: try to call with kwargs and fallback
    try:
        model_result = model_mod.MLP(**mlp_args, lr=float(config["lr"]), weight_decay=1e-4)
    except TypeError:
        model_result = model_mod.MLP(**mlp_args)

    if isinstance(model_result, tuple):
        model, optimizer = model_result
    else:
        model = model_result
        optimizer = None

    model = model.to(device)
    print("[main] model created and moved to device.")

    pool_args = {
        "N_pool": int(config["N_pool"]),
        "M": int(config["M"]),
        "W": int(config["W"]),
        "gamma_r": float(config["gamma_r"]),
        "a0": float(config["a0"]),
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
        train_mod.train(
            model=model,
            dataset=dataset,
            N_pool=int(config["N_pool"]),
            L=int(config["L"]),
            B=int(config["batch_size"]),
            gamma=float(config["gamma"]),
            tau=float(config["tau"]),
            lr=float(config["lr"]),
            epochs=int(config["epochs"]),
            S=int(config["S_cache"]),
            pool_args=pool_args,
            pout_args=pout_args,
            val_dataset=None,
            env_pool=env_pool
        )
    except TypeError:
        # fallback: try positional or simplified signature
        try:
            sig = inspect.signature(train_mod.train)
            print(f"[main] train.train signature: {sig}. Attempting positional fallback.")
        except Exception:
            pass
        # fallback simple attempt; student should adapt if their train API is custom
        train_mod.train(model, dataset, pool_args, save_dir, env_pool)

    print("[main] training finished. Check the save directory for outputs.")


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
