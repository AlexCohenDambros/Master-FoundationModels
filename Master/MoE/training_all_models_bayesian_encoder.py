import subprocess
import os
import json
import optuna
import math
import torch
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from run_experiments_benchmark import run_benchmark_experiment_pipeline
from setup.models.modeling_model_encoder import predict_from_model


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

N_CORES = 10
device  = "cuda"

# ======================================
# PATHS
# ======================================
TRAIN_BASE_PATH = "../benchmark_prepared_train"
VAL_BASE_PATH   = "../benchmark_prepared_val"
TRAINED_MODELS  = "trained_models_benchmark_filtered"

# ======================================
# DATASET CONFIG
# ======================================
DATASETS_TRAIN = {
    "cif_2016_filtered":        {"subfolder": "horizon_12", "horizon": 12},
    "etth_filtered":            {"subfolder": "horizon_36", "horizon": 36},
    "hospital_filtered":        {"subfolder": "horizon_12", "horizon": 12},
    "m3_monthly_filtered":      {"subfolder": "horizon_18", "horizon": 18},
    "m4_monthly_filtered":      {"subfolder": "horizon_18", "horizon": 18},
    "nn5_weekly_filtered":      {"subfolder": "horizon_8",  "horizon": 8},
    "tourism_monthly_filtered": {"subfolder": "horizon_24", "horizon": 24},
    "weather_filtered":         {"subfolder": "horizon_36", "horizon": 36},
}

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
MAX_TRIALS = 50
N_WARMUP   = 3
N_TRIALS   = int(min(MAX_TRIALS, total_combinations))


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


def suggest_hyperparams(trial: optuna.Trial) -> dict:
    params = {}
    for name, cfg in SEARCH_SPACE.items():
        if cfg["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, cfg["choices"])
        elif cfg["type"] == "int":
            params[name] = trial.suggest_int(
                name, cfg["low"], cfg["high"], step=cfg.get("step", 1)
            )
        elif cfg["type"] == "float":
            params[name] = trial.suggest_float(
                name, cfg["low"], cfg["high"], log=cfg.get("log", False)
            )
    return params


def params_signature(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


# ======================================
# DATA LOADING
# ======================================
def load_val_data(dataset_name: str, horizon: int, subfolder: str):
    """Load all series from benchmark_prepared_val and split into train/test tensors."""
    file_path = os.path.join(VAL_BASE_PATH, dataset_name, subfolder, "dataset.jsonl")

    if not os.path.isfile(file_path):
        print(f"  [WARNING] Val data not found: {file_path}", flush=True)
        return None, None

    train_list, test_list = [], []

    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            for _, value in entry.items():
                if len(value) >= horizon + 1:
                    train_list.append(value[:-horizon])
                    test_list.append(value[-horizon:])

    if not train_list:
        print(f"  [WARNING] No valid sequences for {dataset_name}", flush=True)
        return None, None

    tensor_train = torch.tensor(train_list, dtype=torch.float32)
    tensor_test  = torch.tensor(test_list,  dtype=torch.float32)
    return tensor_train, tensor_test


# ======================================
# EVALUATION
# ======================================
def evaluate_dataset(
    model_path: str,
    tensor_train: torch.Tensor,
    tensor_test: torch.Tensor,
    horizon: int,
    top_k: int,
    use_noise: str,
    device: str,
) -> float:
    """Normalize, predict, denormalize, then return mean MAPE across all series."""
    mean_vals = tensor_train.mean(dim=1, keepdim=True)
    std_vals  = tensor_train.std(dim=1, keepdim=True)
    std_vals[std_vals == 0] = 1e-8

    tensor_train_scaled = ((tensor_train - mean_vals) / std_vals).to(device)

    try:
        preds = predict_from_model(
            model_path=model_path,
            series=tensor_train_scaled,
            horizon=horizon,
            top_k=top_k,
            use_noise=use_noise,
            device=device,
        )
    except Exception as e:
        print(f"  [ERROR] predict_from_model failed: {e}", flush=True)
        return float("inf")

    mean_vals = mean_vals.to(device)
    std_vals  = std_vals.to(device)
    preds = torch.clamp(preds.to(device) * std_vals + mean_vals, min=0)

    mapes = []
    for i in range(preds.shape[0]):
        y_true = tensor_test[i].cpu().numpy()
        y_pred = preds[i].cpu().numpy()
        if np.abs(y_true).sum() == 0:
            continue
        mapes.append(mean_absolute_percentage_error(y_true, y_pred) * 100)

    return float(np.mean(mapes)) if mapes else float("inf")


# ======================================
# TRAINING
# ======================================
def build_train_command(
    dataset_path: str,
    horizon: int,
    top_k: int,
    norm: str,
    use_noise: str,
    epochs: int,
    lr: float,
    save_model_path: str,
    device: str,
) -> list:
    return [
        "python", "main_encoder.py",
        "--mode",      "train",
        "--data",      dataset_path,
        "--horizon",   str(horizon),
        "--top_k",     str(top_k),
        "--use_noise", use_noise,
        "--norm",      norm,
        "--epochs",    str(epochs),
        "--lr",        str(lr),
        "--save_path", save_model_path,
        "--device",    device,
    ]


def run_training_for_dataset(
    dataset_name: str,
    dataset_cfg: dict,
    experiment_name: str,
    models_subdir: str,
    top_k: int,
    norm: str,
    use_noise: str,
    epochs: int,
    lr: float,
    device: str,
    data_base_path: str,
) -> tuple:
    """Train one model for a dataset. Returns (model_path, skipped)."""
    horizon     = dataset_cfg["horizon"]
    subfolder   = dataset_cfg["subfolder"]
    dataset_path = os.path.join(data_base_path, dataset_name, subfolder, "dataset.jsonl")

    if not os.path.isfile(dataset_path):
        print(f"  [SKIP] Dataset not found: {dataset_path}", flush=True)
        return None, True

    model_dir = os.path.join(
        TRAINED_MODELS, dataset_name, models_subdir,
        experiment_name, f"horizon_{horizon}",
    )
    os.makedirs(model_dir, exist_ok=True)

    save_model_path = os.path.join(model_dir, f"model_{experiment_name}.pt")

    if os.path.isfile(save_model_path):
        print(f"  [SKIP] Model already exists: {save_model_path}", flush=True)
        return save_model_path, True

    command = build_train_command(
        dataset_path, horizon, top_k, norm,
        use_noise, epochs, lr, save_model_path, device,
    )

    print(
        f"  Training {dataset_name} | horizon={horizon} | {experiment_name}",
        flush=True,
    )

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
        print(f"  [DONE] {dataset_name}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  [FAILED] {dataset_name}: {e.stderr}", flush=True)
        return None, False

    return save_model_path, False


# ======================================
# DUPLICATE DETECTION
# ======================================
_SEEN_STATES = (
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.RUNNING,
    optuna.trial.TrialState.WAITING,
)


# ======================================
# OBJECTIVE FUNCTION
# ======================================
def objective(trial: optuna.Trial) -> float:
    params = suggest_hyperparams(trial)

    sig = params_signature(params)
    for t in trial.study.trials:
        if t.number == trial.number:
            continue
        if t.state not in _SEEN_STATES:
            continue
        if params_signature(t.params) == sig:
            print(
                f"[Trial {trial.number}] Duplicate of trial #{t.number} "
                f"({t.state.name}) — pruning.",
                flush=True,
            )
            raise optuna.exceptions.TrialPruned()

    top_k     = params["top_k"]
    norm      = params["norm"]
    use_noise = params["use_noise"]
    epochs    = params["epochs"]
    lr        = params["lr"]

    experiment_name = build_experiment_name(top_k, norm, use_noise, epochs, lr)

    print(f"\n[Trial {trial.number}] {experiment_name}", flush=True)

    mapes_all = []

    for dataset_name, dataset_cfg in DATASETS_TRAIN.items():
        print(f"\n  Dataset: {dataset_name}", flush=True)

        model_path, _ = run_training_for_dataset(
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            experiment_name=experiment_name,
            models_subdir="train",
            top_k=top_k,
            norm=norm,
            use_noise=use_noise,
            epochs=epochs,
            lr=lr,
            device=device,
            data_base_path=TRAIN_BASE_PATH,
        )

        if model_path is None:
            print(
                f"  [WARNING] Skipping evaluation for {dataset_name} "
                f"(training failed or dataset missing)",
                flush=True,
            )
            mapes_all.append(float("inf"))
            continue

        tensor_train, tensor_test = load_val_data(
            dataset_name, dataset_cfg["horizon"], dataset_cfg["subfolder"]
        )

        if tensor_train is None or tensor_test is None:
            print(f"  [WARNING] No val data for {dataset_name}", flush=True)
            mapes_all.append(float("inf"))
            continue

        mape = evaluate_dataset(
            model_path=model_path,
            tensor_train=tensor_train,
            tensor_test=tensor_test,
            horizon=dataset_cfg["horizon"],
            top_k=top_k,
            use_noise=use_noise,
            device=device,
        )

        print(f"  [{dataset_name}] MAPE = {mape:.4f}%", flush=True)
        mapes_all.append(mape)

    valid_mapes = [m for m in mapes_all if m != float("inf")]
    avg_mape = float(np.mean(valid_mapes)) if valid_mapes else float("inf")

    print(
        f"[Trial {trial.number}] avg MAPE = {avg_mape:.4f}% | params = {params}\n",
        flush=True,
    )
    return avg_mape


# ======================================
# RETRAIN BEST MODEL ON VAL DATA
# ======================================
def retrain_best_model(best_params: dict, experiment_name: str, device: str):
    """Retrain best config on benchmark_prepared_val for each dataset, then run pipeline."""
    top_k     = best_params["top_k"]
    norm      = best_params["norm"]
    use_noise = best_params["use_noise"]
    epochs    = best_params["epochs"]
    lr        = best_params["lr"]

    print(
        f"\n{'='*60}\n  RETRAINING BEST MODEL ON VAL DATA\n{'='*60}",
        flush=True,
    )

    for dataset_name, dataset_cfg in DATASETS_TRAIN.items():
        print(f"\n  Retraining {dataset_name} with best params ...", flush=True)

        model_path, _ = run_training_for_dataset(
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            experiment_name=experiment_name,
            models_subdir="best_model",
            top_k=top_k,
            norm=norm,
            use_noise=use_noise,
            epochs=epochs,
            lr=lr,
            device=device,
            data_base_path=VAL_BASE_PATH,
        )

        if model_path is None:
            print(f"  [WARNING] Retrain failed for {dataset_name}", flush=True)

    print("\n  Running benchmark prediction pipeline ...", flush=True)
    run_benchmark_experiment_pipeline(
        experiment_name=experiment_name,
        trained_models_root=TRAINED_MODELS,
        datasets=DATASETS_TRAIN,
        top_k=top_k,
        use_noise=use_noise == "true",
    )


# ======================================
# BAYESIAN OPTIMIZATION
# ======================================
if __name__ == "__main__":

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_WARMUP,
        multivariate=True,
        seed=42,
        constant_liar=True,
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
        n_jobs=N_CORES,
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
    print(f"  Best metric : {best.value:.4f}%  (avg MAPE)")
    print("  Best params :")
    for k, v in best.params.items():
        print(f"    {k:12s} = {v}")
    print("=" * 60)

    completed = [t for t in study.trials if t.value is not None]
    ranked    = sorted(completed, key=lambda t: t.value or 0.0)

    print(
        f"\n  {'Rank':<5} {'Trial':<7} {'MAPE%':<12} "
        f"{'epochs':<8} {'lr':<10} {'top_k':<7} {'norm':<6} {'noise'}"
    )
    for rank, t in enumerate(ranked, 1):
        p = t.params
        print(
            f"  {rank:<5} #{t.number:<6} {t.value:<12.4f} "
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
    # RETRAIN + PIPELINE WITH BEST CONFIG
    # ======================================
    bp = best.params
    best_name = build_experiment_name(
        bp["top_k"], bp["norm"], bp["use_noise"], bp["epochs"], bp["lr"]
    )

    retrain_best_model(bp, best_name, device)
    print("\nDone. Best models retrained and predictions generated for all datasets.")
