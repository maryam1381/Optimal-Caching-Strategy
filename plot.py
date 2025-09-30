import re
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

# --- What to plot (must match your CSV column suffixes) ---
METRICS_TO_PLOT = ["CVaR_0.05", "mean_utility_mean", "mean_loss_mean"]

# --- Column prefixes for different methods ---
METHOD_PREFIXES = {
    "rl2o":       ["RLO_CVaR_"],
    "meanopt":    ["Mean-Opt_"],
    "popularity": ["Popularity-TopS_"],
}

METHOD_LABELS = {
    "rl2o": "RL2O-CVaR",
    "meanopt": "Mean-Opt",
    "popularity": "Popularity-TopS",
}

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# -----------------------------------------------------------------------------
# This section is updated to match the folder names from your image.
# -----------------------------------------------------------------------------
EXPERIMENTS = {
    "exp1": {
        "regex": re.compile(r"^exp1_gamma_r=([0-9.]+)$", re.IGNORECASE),
        "param_label": "γ_r"
    },
    "exp2": {
        "regex": re.compile(r"^exp2_lambda_true=([0-9.]+)$", re.IGNORECASE),
        "param_label": "λ_true"
    },
    "exp3": {
        "regex": re.compile(r"^exp3_W=([0-9.]+)$", re.IGNORECASE),
        "param_label": "W"
    },
    "exp4": {
        "regex": re.compile(r"^exp4_L=([0-9.]+)$", re.IGNORECASE),
        "param_label": "L"
    }
}


# ------------------------
# Debug helpers / logging
# ------------------------
def dbg_print_dir_contents(path: Path, max_items: int = 50) -> None:
    try:
        if not path.exists():
            print(f"[DEBUG] Path does not exist: {path}")
            return
        if not path.is_dir():
            print(f"[DEBUG] Path is not a directory: {path}")
            return
        items = list(path.iterdir())
        print(f"[DEBUG] Listing of {path} ({len(items)} items):")
        for i, p in enumerate(items[:max_items]):
            typ = "DIR " if p.is_dir() else "FILE"
            print(f"  {i+1:>3}. [{typ}] {p.name}")
        if len(items) > max_items:
            print(f"  ... (+{len(items)-max_items} more)")
    except Exception as e:
        print(f"[DEBUG] Could not list directory {path}: {e}")


def sanitize_base_arg(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s.strip().strip('"').strip("'")


def resolve_base_dir(base_arg: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if base_arg:
        p = Path(base_arg)
        candidates.append(p)
        candidates.append(Path.cwd() / base_arg)

    here = Path(__file__).resolve().parent
    candidates.extend([
        here / "results",
        here.parent / "results",
        here.parent.parent / "results",
        Path("results"),
    ])
    candidates.append(Path(r"D:\Github\Optimal-Caching-Strategy\results"))

    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                return c.resolve()
        except Exception:
            continue
    return None


# ------------------------
# CSV reading and parsing
# ------------------------
def _find_prefixed_column(columns: List[str], prefixes: List[str], metric: str) -> Optional[str]:
    lower_cols = {c.lower(): c for c in columns}
    for p in prefixes:
        target = (p + metric).lower()
        if target in lower_cols:
            return lower_cols[target]
    return None


def read_timeseries(csv_path: Path) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
    out: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed reading {csv_path}: {e}")
        return out

    epoch_col = None
    for c in ["epoch", "Epoch", "iter", "step"]:
        if c in df.columns:
            epoch_col = c
            break
    if epoch_col is None:
        df = df.reset_index().rename(columns={"index": "epoch"})
        epoch_col = "epoch"

    columns = list(df.columns)
    found_any = False

    for method, prefixes in METHOD_PREFIXES.items():
        for metric in METRICS_TO_PLOT:
            col = _find_prefixed_column(columns, prefixes, metric)
            if col is None:
                continue
            ts = df[[epoch_col, col]].copy()
            ts.columns = ["epoch", "value"]
            ts["epoch"] = pd.to_numeric(ts["epoch"], errors="coerce")
            ts["value"] = pd.to_numeric(ts["value"], errors="coerce")
            ts = ts.dropna().sort_values("epoch")
            if not ts.empty:
                out.setdefault(method, []).append((metric, ts))
                found_any = True

    if not found_any:
        print(f"[WARN] No recognized method-prefixed columns in {csv_path.name}")
    return out


def average_across_seeds(ts_list: List[pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for i, df in enumerate(ts_list):
        df = df.set_index("epoch").rename(columns={"value": f"v{i}"})
        merged = df if merged is None else merged.join(df, how="outer")
    merged = merged.sort_index()
    mean_series = merged.mean(axis=1, skipna=True).to_frame(name="value")
    mean_series.reset_index(inplace=True)
    return mean_series


# ------------------------
# Collection & plotting
# ------------------------
def collect_data(base_dir: Path, exp_config: Dict[str, Any]):
    """Generalized data collection function for a given experiment."""
    exp_regex = exp_config["regex"]
    data: Dict[str, Dict[str, Dict[float, pd.DataFrame]]] = {m: {} for m in METRICS_TO_PLOT}
    exp_dirs: List[Path] = []

    try:
        for sub in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
            m = exp_regex.match(sub.name)
            if not m:
                continue
            exp_dirs.append(sub)
    except Exception as e:
        print(f"[ERROR] Could not iterate base dir {base_dir}: {e}")
        return data, []

    if not exp_dirs:
        print(f"[WARN] No folders matching pattern '{exp_regex.pattern}' under: {base_dir}")
        return data, []

    print(f"[INFO] Found {len(exp_dirs)} experiment folders:")
    for d in exp_dirs:
        print(f"       - {d.name}")

    for sub in exp_dirs:
        try:
            param_value = float(exp_regex.match(sub.name).group(1))
        except Exception:
            print(f"[WARN] Could not parse parameter value from folder name: {sub.name}")
            continue

        ckpt = sub / "checkpoints"
        if not ckpt.exists():
            print(f"[WARN] Missing folder: {ckpt}")
            continue

        csvs = sorted(ckpt.glob("*.csv"))
        if not csvs:
            print(f"[WARN] No CSV files in: {ckpt}")
            continue

        bucket: Dict[str, Dict[str, List[pd.DataFrame]]] = {}
        for csv_p in csvs:
            method_to_series = read_timeseries(csv_p)
            for method_key, series_list in method_to_series.items():
                for metric, ts in series_list:
                    bucket.setdefault(method_key, {}).setdefault(metric, []).append(ts)

        for method_key, per_metric in bucket.items():
            for metric, ts_list in per_metric.items():
                if not ts_list:
                    continue
                mean_ts = average_across_seeds(ts_list)
                data.setdefault(metric, {}).setdefault(method_key, {})[param_value] = mean_ts

    all_param_values = set()
    for metric in data:
        for method in data[metric]:
            all_param_values.update(data[metric][method].keys())
    params_sorted = sorted(all_param_values)
    return data, params_sorted


def plot_one(metric: str,
             method: str,
             param_values: List[float],
             series_by_param: Dict[float, pd.DataFrame],
             out_dir: Path,
             exp_name: str,
             param_label: str):
    if not param_values:
        print(f"[WARN] No parameter values to plot for {metric} / {method}")
        return

    fig = plt.figure(figsize=(7, 4), dpi=140)
    ax = fig.add_subplot(111)
    drew_any = False

    for p_val in param_values:
        ts = series_by_param.get(p_val)
        if ts is None or ts.empty:
            print(f"[WARN] No data for {param_label}={p_val} on {metric}/{method}")
            continue
        ax.plot(ts["epoch"].values, ts["value"].values, marker='o', linewidth=1.2, label=f"{param_label} = {p_val:g}")
        drew_any = True

    if not drew_any:
        plt.close(fig)
        print(f"[WARN] Skipping plot: no curves for {metric}/{method}")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(METHOD_LABELS.get(method, method))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc="best")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{exp_name}_{metric}_{method}_epochs.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVE] {out}")


def plot_combined_one(metric: str,
                      param_values: List[float],
                      data_for_metric: Dict[str, Dict[float, pd.DataFrame]],
                      out_dir: Path,
                      exp_name: str,
                      param_label: str):
    """
    One figure with 3 subplots: Mean-Opt | Popularity-TopS | RL2O-CVaR
    """
    methods_in_order = ["meanopt", "popularity", "rl2o"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=140, sharey=True)
    fig.suptitle(f"{exp_name.upper()}: {metric}", fontsize=16)

    for ax, method in zip(axes, methods_in_order):
        series_by_param = data_for_metric.get(method, {})
        drew_any = False
        for p_val in param_values:
            ts = series_by_param.get(p_val)
            if ts is None or ts.empty:
                continue
            ax.plot(ts["epoch"].values, ts["value"].values, marker='o', linewidth=1.2, label=f"{param_label} = {p_val:g}")
            drew_any = True
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle='--', alpha=0.3)
        if method == methods_in_order[0]:
            ax.set_ylabel(metric)
        if drew_any:
            ax.legend(loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{exp_name}_{metric}_ALL_epochs.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    fig.savefig(out)
    plt.close(fig)
    print(f"[SAVE] {out} (combined 3-in-1)")

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot learning curves for multiple experiments from results directory.")
    ap.add_argument("--base", type=str, default=None, help="Path to 'results' directory.")
    ap.add_argument("--out", type=str, default=None, help="Base output directory for figures (e.g., 'figures').")
    args = ap.parse_args()

    raw_base = sanitize_base_arg(args.base)
    base_dir = resolve_base_dir(raw_base)

    if base_dir is None:
        print("[ERROR] Could not locate the 'results' directory.")
        print("  Please specify the path to your 'results' folder, for example:")
        print(r"    python plot.py --base D:\Github\Optimal-Caching-Strategy\results")
        sys.exit(1)

    print(f"[INFO] Using base results directory: {base_dir}")
    
    # Determine base output directory
    if args.out:
        base_out_dir = Path(args.out.strip().strip('"').strip("'"))
    else:
        # Default to a 'figures' directory alongside the 'results' directory
        base_out_dir = base_dir.parent / "figures"
    
    print(f"[INFO] Figures will be saved in subdirectories under: {base_out_dir.resolve()}")

    # --- Loop through all defined experiments ---
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"\n{'='*60}\n[INFO] Processing experiment: {exp_name.upper()}\n{'='*60}")
        
        # Collect data for the current experiment
        data, param_values = collect_data(base_dir, exp_config)
        
        if not param_values:
            print(f"[WARN] No data found for {exp_name}. Skipping.")
            continue
            
        # Define a specific output directory for this experiment's plots
        exp_out_dir = base_out_dir / f"figs_{exp_name}"
        exp_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving {exp_name} plots to: {exp_out_dir}")

        # Plot individual per-method figures
        for metric in METRICS_TO_PLOT:
            if metric not in data:
                print(f"[WARN] Metric '{metric}' not present in any CSV for {exp_name}")
                continue
            for method in ["rl2o", "meanopt", "popularity"]:
                series_by_param = data[metric].get(method, {})
                if not series_by_param:
                    print(f"[WARN] No data for method '{method}' (metric: {metric}) in {exp_name}")
                    continue
                plot_one(metric, method, param_values, series_by_param, exp_out_dir, exp_name, exp_config["param_label"])

        # Plot combined 3-in-1 figures
        for metric in METRICS_TO_PLOT:
            if metric not in data:
                continue
            plot_combined_one(metric, param_values, data[metric], exp_out_dir, exp_name, exp_config["param_label"])

    print("\n[INFO] All experiments processed. Done.")

if __name__ == "__main__":
    main()