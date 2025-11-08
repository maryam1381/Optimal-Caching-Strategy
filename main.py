# main.py
# #!/usr/bin/env python3
"""
main.py — Top-level experiment orchestration for the Digital-Twin-assisted
CVaR-robust D2D caching project.

Usage (from repository root):
    python main.py --help
...
"""

from __future__ import annotations
import argparse
import json

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
# from plot import generate_plots, plot_training_curves

# --- UPDATED IMPORT ---
from src.final_comparison import run_final_comparison
from plot import plot_training_curves
from src.losses import batch_utility_from_env_samples, project_capped_simplex
from src import env_pool as env_pool_mod
from src import model as model_mod
from src import train as train_mod
from src import utils as utils_mod
from src.model_eval import evaluate_model # <-- IMPORTED for validation

# Make repo root and src discoverable even if user runs script from another cwd
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_PATH))

# -------------------------
# Default experiment config
# -------------------------
DEFAULT_CONFIG = {
    "M": 100, 
    "W": 200, 
    "gamma_r": 0.8, 
    "a0_p": 1.0,
    "a0_lambda": 1.0, 
    "b0": 1.0, 
    "A_user": 10.0, 
    "lambda_true": 2.5,
    "N_pool": 40000, 
    "dataset_size": 10000, 
    "test_set_fraction": 0.2,   # <-- ADDED: Fraction for test set
    "val_set_fraction": 0.1,    # <-- ADDED: Fraction for validation set
    "batch_size": 64, 
    "L": 256,  # sample size 
    "gamma": 0.05, 
    "tau": 15.0, 
    "lr": 5e-3,         
    "epochs": 100,  
    "S_cache": 20, 
    "hidden_small": False,  # Use larger hidden layers
    "dropout": 0.1,  # Added dropout to prevent overfitting
    "seed": 42,  
    "save_dir": "results/",
}


# -------------------------
# Utility: build synthetic measurement dataset q
# -------------------------
def build_synthetic_dataset_from_pool(pool: tuple, dataset_size: int, M: int, W: int, A_user: float,
                                      lambda_true: float, rng: np.random.Generator):
    # pool is (P_pool_array, LAM_pool_array)
    P_pool, LAM_pool = pool
    pool_size = len(P_pool)
    dataset = np.zeros((dataset_size, M + 1), dtype=np.float32)
    for i in range(dataset_size):
        idx = int(rng.integers(0, pool_size))
        p_true = P_pool[idx]
        p_true = np.asarray(p_true, dtype=np.float64)
        
        # observed request counts (Multinomial)
        n = rng.multinomial(W, p_true)
        # observed number of users (Poisson)
        K = int(rng.poisson(lam=lambda_true * A_user))
        
        # Normalize/process features
        n_norm = n / (W + 1e-12)
        K_norm = K / max(1.0, A_user)
        
        # Ensure K_norm is a scalar float before assignment
        dataset[i, :M] = n_norm.astype(np.float32)
        dataset[i, M] = float(K_norm)
        
    return dataset


def main(config: Dict[str, Any]):
    rng = utils_mod.set_seed(int(config["seed"]))
    device = utils_mod.get_device(prefer_cuda=True)
    save_dir = Path(config["save_dir"])
    utils_mod.ensure_dir(str(save_dir))

    # Save the master config file for this experiment
    utils_mod.save_json(str(save_dir / "config.json"), config)
    
    # 1) Load config (omitted file ops for brevity)
    
    # 2) Build Pool and Full Dataset
    env_pool = env_pool_mod.build_env_pool_simulated(
        N_pool=int(config["N_pool"]), M=int(config["M"]), W=int(config["W"]),
        zipf_exponent=float(config["gamma_r"]), a0_p=float(config.get("a0_p", 1.0)),
        a0_lambda=float(config.get("a0_lambda", 1.0)), b0_lambda=float(config.get("b0", 1.0)),
        A_obs=float(config.get("A_user", 1.0)), lambda_true=float(config.get("lambda_true", 1.0)),
        rng=rng, as_torch=False
    )
    dataset_rng = np.random.default_rng(int(config["seed"]) + 1)
    dataset = build_synthetic_dataset_from_pool(
        pool=env_pool, dataset_size=int(config["dataset_size"]), M=int(config["M"]),
        W=int(config["W"]), A_user=float(config["A_user"]),
        lambda_true=float(config["lambda_true"]), rng=dataset_rng
    )
    
    # --- Shuffle and Split Dataset ---
    rng.shuffle(dataset)
    
    test_fraction = float(config.get("test_set_fraction", 0.2))
    val_fraction = float(config.get("val_set_fraction", 0.1))
    
    if (test_fraction + val_fraction) >= 1.0:
        raise ValueError("test_set_fraction and val_set_fraction must sum to less than 1.0")

    dataset_size = len(dataset)
    test_size = int(dataset_size * test_fraction)
    val_size = int(dataset_size * val_fraction)
    
    test_dataset_q = dataset[:test_size]
    val_dataset_q = dataset[test_size : test_size + val_size]
    train_dataset_q = dataset[test_size + val_size :]
    
    print(f"[main] Dataset split: {len(train_dataset_q)} train, {len(val_dataset_q)} val, {len(test_dataset_q)} test.")
    if len(train_dataset_q) == 0:
        raise ValueError("Training set is empty. Adjust dataset_size or fractions.")
    if len(val_dataset_q) == 0:
        print("WARNING: Validation set is empty. Training will not have validation metrics.")
    if len(test_dataset_q) == 0:
        print("WARNING: Test set is empty. Final comparison will not run on any data.")
    
    
    # 3) Build model (MLP) & Get Optimizer
    model ,optimizer = model_mod.create_mlp(M=int(config["M"]), 
                                            use_layernorm=True,
                                                dropout=float(config["dropout"]), 
                                                lr=float(config["lr"]), 
                                                weight_decay=1e-4)

    model = model.to(device)
    print(f"[main] Model created and moved to device: {device}")

    # 4) Setup TrainConfig with all experiment parameters
    train_config = train_mod.TrainConfig(
        N_pool=int(config["N_pool"]), 
        L=int(config["L"]), 
        batch_size=int(config["batch_size"]),
        gamma_tail=float(config["gamma"]), 
        tau=float(config["tau"]), 
        epochs=int(config["epochs"]),
        lr=float(config["lr"]), 
        device=device.type, 
        checkpoint_dir=str(save_dir/"checkpoints"), 
        verbose=True,
        zipf_exponent=float(config["gamma_r"]), 
        lambda_true=float(config["lambda_true"]), 
        W=int(config["W"]), 
        seed=int(config["seed"])
    )

    # 5) Call the training routine
    print("[main] Calling training routine...")
    train_mod.train_loop(
        model=model,
        optimizer=optimizer, 
        env_pool=env_pool, 
        dataset_q=train_dataset_q,      # Pass train set
        S=int(config["S_cache"]), 
        compute_utility_fn=batch_utility_from_env_samples,
        project_fn=project_capped_simplex, 
        config=train_config, 
        rng=rng,
        val_dataset_q=val_dataset_q,    # Pass val set
        eval_fn=evaluate_model          # Pass the eval function
    )
    print("[main] Training finished. Check the save directory for outputs.")

    # --- UPDATED FINAL COMPARISON BLOCK ---
    print("\n[main] --- Starting Final Comparison ---")
    
    # The model is now trained and in eval mode (or should be set to it)
    model.eval() 
    
    # Convert env_pool tuple of arrays (P, LAM) to a list of tuples [(p1, l1), (p2, l2), ...]
    P_pool, LAM_pool = env_pool
    print(f"[main] Converting env_pool (P:{P_pool.shape}, LAM:{LAM_pool.shape}) to list...")
    env_pool_list = list(zip(P_pool, LAM_pool))
    print(f"[main] ... converted to list of {len(env_pool_list)} scenarios.")

    # Call our new comparison function
    # It now returns a DataFrame directly.
    df_results = run_final_comparison(
        model=model,
        test_dataset_q=test_dataset_q,        # <-- Evaluate on the test set
        env_pool_list=env_pool_list,      # Pass the list of scenarios
        project_fn=project_capped_simplex,  # Pass the projection function
        S=float(config["S_cache"]),
        gamma=float(config["gamma"]),
        device=device,
        rng=rng,
        n_env_eval_model=500,       # Number of env samples per q for the model
        n_env_eval_baseline=5000    # Number of env samples for the (single) Top-S policy
    )

    print("\n[main] --- Comparison Results ---")
    print(df_results)
    
    # Save the results to a CSV file in the experiment's save directory
    results_path = save_dir / "comparison_results.csv"
    try:
        df_results.to_csv(results_path)
        print(f"Comparison results saved to {results_path}")
    except Exception as e:
        print(f"ERROR: Could not save comparison results to {results_path}. Error: {e}")
    print("----------------------------------\n")
    # --- END UPDATED BLOCK ---


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


def run_full_sweep(base_cfg: Dict[str, Any]):
    """
    Run the 4 experiments shown in the figure:
      Exp1: gamma_r ∈ {0.6, 1.0, 1.5, 2.0}
      Exp2: lambda_true ∈ {0.5, 1.5, 5.0}
      Exp3: K ∈ {5, 20, 100}  and  W ∈ {20, 100, 500}
      Exp4: L ∈ {64, 128, 256}
    Notes:
      • In our code, expected user count K is Poisson(mean=lambda_true * A_user).
        For Exp3 we control the *mean* ≈ K by setting lambda_true = 1.0 and A_user = K.
      • W maps directly to --W (measurement window).
    """
    import copy
    run_id = 0
    base_save = Path(base_cfg["save_dir"]).resolve()

    def bump_seed(cfg, add):
        cfg["seed"] = int(cfg.get("seed", 42)) + int(add)
        return cfg

    # ---------- Exp 1: Zipf exponent γ_r
    for gr in [0.6, 1.0, 1.5, 2.0]:
        cfg = copy.deepcopy(base_cfg)
        cfg["gamma_r"] = float(gr)
        cfg["save_dir"] = str(base_save / f"exp1_gamma_r={gr}")
        bump_seed(cfg, run_id); run_id += 1
        print(f"\n[SWEEP] Exp1 lambda_r={gr} -> save_dir={cfg['save_dir']}")
        main(cfg)

    # ---------- Exp 2: User density λ (lambda_true)
    for lam in [0.5, 1.5, 5.0]:
        cfg = copy.deepcopy(base_cfg)
        cfg["lambda_true"] = float(lam)
        cfg["save_dir"] = str(base_save / f"exp2_lambda_true={lam}")
        bump_seed(cfg, run_id); run_id += 1
        print(f"\n[SWEEP] Exp2 lambda={lam} -> save_dir={cfg['save_dir']}")
        main(cfg)

    # ---------- Exp 3: Grid over K and W
    # We set lambda_true=1.0 and A_user=K to make E[K]≈K.
    # for K_mean in [5, 20, 100]:
    # ---------- Exp 3: Only sweep W
    for W_val in [20, 100, 500]:
        cfg = copy.deepcopy(base_cfg)
        cfg["lambda_true"] = 1.0    # keep fixed
        cfg["W"] = int(W_val)
        cfg["save_dir"] = str(base_save / f"exp3_W={W_val}")
        bump_seed(cfg, run_id); run_id += 1
        print(f"\n[SWEEP] Exp3 W={W_val} -> save_dir={cfg['save_dir']}")
        main(cfg)


    # ---------- Exp 4: Number of posterior draws L
    for L_val in [64, 128, 256]:
        cfg = copy.deepcopy(base_cfg)
        cfg["L"] = int(L_val)
        cfg["save_dir"] = str(base_save / f"exp4_L={L_val}")
        bump_seed(cfg, run_id); run_id += 1
        print(f"\n[SWEEP] Exp4 L={L_val} -> save_dir={cfg['save_dir']}")
        main(cfg)



if __name__ == "__main__":
    args = parse_args()
    # load JSON config if provided
    cfg = DEFAULT_CONFIG.copy()
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        cfg.update(cfg_json)

    # Update config with any CLI overrides
    cfg.update({
        "save_dir": args.save_dir, "seed": args.seed, "epochs": args.epochs,
        "N_pool": args.N_pool, "dataset_size": args.dataset_size, "M": args.M,
        "W": args.W, "L": args.L, "batch_size": args.batch_size, "lr": args.lr,
        "gamma": args.gamma, "S_cache": args.S_cache, "lambda_true": args.lambda_true,
    })
    
    try:
        # --- STAGE 1: Run the full training sweep ---
        # This will now also run the comparison at the end of each 'main' call
        print("--- [STAGE 1/2] Starting full training and evaluation sweep... ---")
        run_full_sweep(cfg)
        print("--- [STAGE 1/2] Training and evaluation sweep complete. ---")
        
        # --- STAGE 2: Plot training curves ---
        print("\n--- [STAGE 2/2] Generating training plots... ---")
        # Pass the *base* save directory
        plot_training_curves(results_folder=cfg["save_dir"])
        print("--- [STAGE 2/2] Plot generation complete. ---")

        print(f"\n All steps finished successfully.")
        print(f"Final results are saved in: {Path(cfg['save_dir']).resolve()}")
        print(f"Final plots are saved in: {REPO_ROOT / 'figures'}")

    except Exception as e:
        print(f"\n ERROR: An error occurred during the workflow: {e}")
        import traceback
        traceback.print_exc()


# if __name__ == "__main__":
#     args = parse_args()
#     # load JSON config if provided
#     cfg = DEFAULT_CONFIG.copy()
#     if args.config is not None:
#         with open(args.config, "r", encoding="utf-8") as f:
#             cfg_json = json.load(f)
#         cfg.update(cfg_json)

#     # Update config with any CLI overrides
#     cfg.update({
#         "save_dir": args.save_dir, "seed": args.seed, "epochs": args.epochs,
#         "N_pool": args.N_pool, "dataset_size": args.dataset_size, "M": args.M,
#         "W": args.W, "L": args.L, "batch_size": args.batch_size, "lr": args.lr,
#         "gamma": args.gamma, "S_cache": args.S_cache, "lambda_true": args.lambda_true,
#     })

#     try:
#         # --- STAGE 1: Run the full training sweep ---
#         print("--- [STAGE 1/3] Starting full training sweep... ---")
#         run_full_sweep(cfg)
#         print("--- [STAGE 1/3] Training sweep complete. ---")
#         plot_training_curves()

#         # --- STAGE 2: Run the final evaluation ---
#         # print("\n--- [STAGE 2/3] Starting final evaluation... ---")
#         # run_all_evaluations(results_root="results")
#         # print("--- [STAGE 2/3] Evaluation complete. ---")

#         # # --- STAGE 3: Generate all plots ---
#         # print("\n--- [STAGE 3/3] Generating all plots... ---")
#         # generate_plots(results_folder="results")
#         # print("--- [STAGE 3/3] Plot generation complete. ---")

#         # print(f"\n All steps finished successfully.")
#         # print(f"Final plots are saved in: {REPO_ROOT / 'figures'}")

#     except Exception as e:
#         print(f"\n ERROR: An error occurred during the workflow: {e}")
#         import traceback
#         traceback.print_exc()

