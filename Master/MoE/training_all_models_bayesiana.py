import subprocess
import os
import re
import json
import optuna
from joblib import Parallel, delayed
import math
from run_experiments import run_full_experiment_pipeline


def compute_total_search_space(search_space):
    total = 1
    for cfg in search_space.values():
        if cfg["type"] == "categorical":
            total *= len(cfg["choices"])
        elif cfg["type"] == "int":
            step = cfg.get("step", 1)
            total *= ((cfg["high"] - cfg["low"]) // step) + 1
        elif cfg["type"] == "float":
            return math.inf
    return total

# ======================================
# GENERAL CONFIGURATION
# ======================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

N_CORES  = 1
MAX_TRIALS = 50
N_WARMUP = 3

base_path   = "../all_datasets_global_by_years_test_diff_series"
HORIZONS    = [12]
MIN_YEAR    = 2024
MAX_YEAR    = 2024
debug_state = None
device      = "cpu"

# ======================================
# SEARCH SPACE
# ======================================
SEARCH_SPACE = {
    "top_k":     {"type": "categorical", "choices": [2]},
    "norm":      {"type": "categorical", "choices": ["std"]},
    "use_noise": {"type": "categorical", "choices": ["true"]},
    "epochs":    {"type": "categorical", "choices": [30, 60, 100]},
    "lr":        {"type": "categorical", "choices": [1e-4, 1e-3, 1e-5]},
}

total_combinations = compute_total_search_space(SEARCH_SPACE)
N_TRIALS = min(MAX_TRIALS, total_combinations)

# ======================================
# HELPERS
# ======================================
def build_experiment_name(top_k, norm, use_noise, epochs, lr) -> str:
    return (
        f"topk_{top_k}"
        f"_norm_{norm}"
        f"_noise_{use_noise}"
        f"_ep_{epochs}"
        f"_lr_{lr}"
    )


def is_model_complete(save_model_path: str) -> bool:
    return os.path.isfile(save_model_path)


def suggest_hyperparams(trial: optuna.Trial) -> dict:
    params = {}
    for name, cfg in SEARCH_SPACE.items():
        if cfg["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, cfg["choices"])
        elif cfg["type"] == "int":
            params[name] = trial.suggest_int(name, cfg["low"], cfg["high"], step=cfg.get("step", 1))
        elif cfg["type"] == "float":
            params[name] = trial.suggest_float(name, cfg["low"], cfg["high"], log=cfg.get("log", False))
    return params


def collect_metrics_from_saved_files(expected_paths: list) -> float:
    """Reads only the _metrics.json files this trial was expected to generate."""
    metrics = []
    for model_path in expected_paths:
        metrics_path = model_path.replace(".pt", "_metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        try:
            with open(metrics_path) as fp:
                data = json.load(fp)
            val = data.get("val_loss")
            if val is not None:
                metrics.append(float(val))
        except Exception as e:
            print(f"  [WARNING] Could not read {metrics_path}: {e}", flush=True)

    if not metrics:
        print("  [WARNING] No _metrics.json files found — returning inf.", flush=True)
        return float("inf")

    avg = sum(metrics) / len(metrics)
    print(f"  Metrics collected: {len(metrics)} files | avg val_loss = {avg:.6f}", flush=True)
    return avg


# ======================================
# SINGLE JOB TRAINING
# ======================================
def run_training(command, excluded_state, year, horizon,
                 top_k, norm, dataset_path, device):
    print(
        f"Training: excluding={excluded_state} | year={year} | "
        f"horizon={horizon} | top_k={top_k} | norm={norm} | device={device}",
        flush=True,
    )
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"  Done | excluding={excluded_state} | year={year} | horizon={horizon}", flush=True)
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED | excluding={excluded_state} | year={year} | horizon={horizon}", flush=True)
        print(e.stderr, flush=True)


# ======================================
# BUILD JOB LIST
# ======================================
def build_jobs(trained_models_root, experiment_name, top_k, norm, use_noise, epochs, lr):
    jobs           = []
    skipped        = 0
    expected_paths = []

    for HORIZON in HORIZONS:
        horizon_base_path = os.path.join(base_path, f"horizon_{HORIZON}")
        if not os.path.exists(horizon_base_path):
            continue

        for excluded_state_folder in os.listdir(horizon_base_path):
            if not excluded_state_folder.startswith("excluding_"):
                continue

            excluded_state = excluded_state_folder.replace("excluding_", "")
            if debug_state is not None and excluded_state != debug_state:
                continue

            folder_path = os.path.join(horizon_base_path, excluded_state_folder)
            if not os.path.isdir(folder_path):
                continue

            state_model_dir = os.path.join(
                trained_models_root, f"horizon_{HORIZON}", excluded_state_folder
            )
            os.makedirs(state_model_dir, exist_ok=True)

            for dataset_file in os.listdir(folder_path):
                if not dataset_file.endswith(".jsonl"):
                    continue

                match = re.search(r"dataset_(\d{4})\.jsonl", dataset_file)
                if not match:
                    continue

                year = int(match.group(1))
                if year < MIN_YEAR or year > MAX_YEAR:
                    continue

                dataset_path    = os.path.join(folder_path, dataset_file)
                save_model_path = os.path.join(
                    state_model_dir,
                    f"model_{experiment_name}_{year}.pt",
                )

                expected_paths.append(save_model_path)

                if is_model_complete(save_model_path):
                    skipped += 1
                    continue

                command = [
                    "python", "main_test.py",
                    "--mode",      "train",
                    "--data",      dataset_path,
                    "--horizon",   str(HORIZON),
                    "--top_k",     str(top_k),
                    "--use_noise", use_noise,
                    "--norm",      norm,
                    "--epochs",    str(epochs),
                    "--lr",        str(lr),
                    "--save_path", save_model_path,
                    "--device",    device,
                ]

                jobs.append((
                    command, excluded_state, year, HORIZON,
                    top_k, norm, dataset_path, device,
                ))

    return jobs, skipped, expected_paths


# ======================================
# OBJECTIVE FUNCTION
# ======================================
def objective(trial: optuna.Trial) -> float:
    params    = suggest_hyperparams(trial)
    top_k     = params["top_k"]
    norm      = params["norm"]
    use_noise = params["use_noise"]
    epochs    = params["epochs"]
    lr        = params["lr"]

    experiment_name     = build_experiment_name(top_k, norm, use_noise, epochs, lr)
    trained_models_root = os.path.join("trained_models_test", experiment_name)
    os.makedirs(trained_models_root, exist_ok=True)

    jobs, skipped, expected_paths = build_jobs(
        trained_models_root, experiment_name,
        top_k, norm, use_noise, epochs, lr
    )

    print(
        f"\n[Trial {trial.number}] {experiment_name}\n"
        f"  Jobs to run : {len(jobs)}\n"
        f"  Skipped     : {skipped} (model already exists)",
        flush=True,
    )

    if jobs:
        Parallel(n_jobs=N_CORES, backend="loky", verbose=5)(
            delayed(run_training)(cmd, es, yr, hz, tk, nm, dp, dv)
            for cmd, es, yr, hz, tk, nm, dp, dv in jobs
        )

    metric = collect_metrics_from_saved_files(expected_paths)
    print(f"[Trial {trial.number}] metric = {metric:.6f} | params = {params}\n", flush=True)
    return metric


# ======================================
# BAYESIAN OPTIMIZATION
# ======================================
if __name__ == "__main__":

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_WARMUP,
        multivariate=True,
        seed=42,
    )

    storage = optuna.storages.RDBStorage(
        url="sqlite:///bayesian_search.db",
        heartbeat_interval=60,
    )

    study = optuna.create_study(
        study_name="timeseries_bayesian_search",
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    # ======================================
    # FINAL REPORT
    # ======================================
    best = study.best_trial

    print("\n" + "=" * 60)
    print("  BAYESIAN OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"  Best trial  : #{best.number}")
    print(f"  Best metric : {best.value:.6f}")
    print("  Best params :")
    for k, v in best.params.items():
        print(f"    {k:12s} = {v}")
    print("=" * 60)

    ranked = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
    )

    print(f"\n  {'Rank':<5} {'Trial':<7} {'Metric':<12} {'epochs':<8} {'lr':<10} {'top_k':<7} {'norm':<6} {'noise'}")
    for rank, t in enumerate(ranked, 1):
        p = t.params
        print(
            f"  {rank:<5} #{t.number:<6} {t.value:<12.6f} "
            f"{p.get('epochs',''):<8} {p.get('lr',''):<10} "
            f"{p.get('top_k',''):<7} {p.get('norm',''):<6} {p.get('use_noise','')}"
        )

    summary = {
        "best_trial":  best.number,
        "best_value":  best.value,
        "best_params": best.params,
        "all_trials_ranked": [
            {"rank": rank, "number": t.number, "value": t.value, "params": t.params}
            for rank, t in enumerate(ranked, 1)
        ],
    }
    with open("bayesian_search_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nFull results saved to bayesian_search_results.json")

    # ======================================
    # PREDICTION — BEST TRIAL ONLY
    # ======================================
    bp        = best.params
    best_name = build_experiment_name(
        bp["top_k"], bp["norm"], bp["use_noise"], bp["epochs"], bp["lr"]
    )

    print(f"\nRunning final prediction pipeline with best params: {best_name}")
    run_full_experiment_pipeline(
        path_trained_models=os.path.join("trained_models_test", best_name),
        experiment_name="model_" + best_name,
        top_k=bp["top_k"],
        use_noise=bp["use_noise"] == "true",
    )

    print("\nDone. Predictions generated for best trial only.")