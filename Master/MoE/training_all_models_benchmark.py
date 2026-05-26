import subprocess
import os
from itertools import product
from joblib import Parallel, delayed
from run_experiments_benchmark import run_full_experiment_pipeline

# ======================================
# GENERAL CONFIGURATION
# ======================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

N_CORES = 20

base_path = "../benchmark_prepared_train"

device = "cuda"

# ======================================
# DATASET REGISTRY
# ======================================
DATASETS = {
    # "cif_2016_filtered":        {"subfolder": "horizon_12",  "horizon": 12, "context_length": 96},
    # "etth_filtered":            {"subfolder": "horizon_36",  "horizon": 36, "context_length": 102},
    # "hospital_filtered":        {"subfolder": "horizon_12",  "horizon": 12, "context_length": 60},
    # "m3_monthly_filtered":      {"subfolder": "horizon_18",  "horizon": 18, "context_length": 98},
    # "m4_monthly_filtered":      {"subfolder": "horizon_18",  "horizon": 18, "context_length": 51},
    # "nn5_weekly_filtered":      {"subfolder": "horizon_8",   "horizon": 8,  "context_length": 97},
    # "tourism_monthly_filtered": {"subfolder": "horizon_24",  "horizon": 24, "context_length": 285},
    # "weather_filtered":         {"subfolder": "horizon_36",  "horizon": 36, "context_length": 454},
    "fred_md_filtered":           {"subfolder": "horizon_12",  "horizon": 12, "context_length": 704},
    "m5_filtered":                {"subfolder": "horizon_28",  "horizon": 28, "context_length": 1913},
}

# ======================================
# HYPERPARAMETER GRID
# ======================================
TOP_K_LIST = [2]
NORM_LIST = ["std"]
USE_NOISE_LIST = [True]
EPOCHS_LIST = [30, 60, 100]
LR_LIST = [1e-4, 1e-3, 1e-5]

# ======================================
# HELPERS
# ======================================
def build_experiment_name(top_k, norm, use_noise, epochs, lr):
    return (
        f"topk_{top_k}"
        f"_norm_{norm}"
        f"_noise_{use_noise}"
        f"_ep_{epochs}"
        f"_lr_{lr}"
    )


def is_model_complete(save_model_path: str) -> bool:
    return os.path.isfile(save_model_path)


def run_training(command, dataset_name, horizon, context_length, top_k, norm, dataset_path, device):
    print(
        f"Training: dataset={dataset_name} | "
        f"horizon={horizon} | "
        f"context={context_length} | "
        f"top_k={top_k} | "
        f"norm={norm} | "
        f"data={dataset_path} | "
        f"device={device}",
        flush=True
    )

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

        print(
            f"✔ DONE | dataset={dataset_name} | horizon={horizon}",
            flush=True
        )

        if result.stdout:
            print(result.stdout, flush=True)

        if result.stderr:
            print(result.stderr, flush=True)

    except subprocess.CalledProcessError as e:
        print(
            f"✖ FAILED | dataset={dataset_name} | horizon={horizon}",
            flush=True
        )
        print(e.stderr, flush=True)


# ======================================
# BUILD JOB LIST (GRID SEARCH)
# ======================================
EXPERIMENTS = list(product(
    TOP_K_LIST,
    NORM_LIST,
    USE_NOISE_LIST,
    EPOCHS_LIST,
    LR_LIST
))

jobs = []
skipped = 0

for top_k, norm, use_noise, epochs, lr in EXPERIMENTS:

    experiment_name = build_experiment_name(top_k, norm, use_noise, epochs, lr)

    trained_models_root = os.path.join("trained_models_benchmark", experiment_name)
    os.makedirs(trained_models_root, exist_ok=True)

    for dataset_name, cfg in DATASETS.items():

        horizon = cfg["horizon"]
        dataset_path = os.path.join(base_path, dataset_name, cfg["subfolder"], "dataset.jsonl")

        if not os.path.isfile(dataset_path):
            print(f"[WARN] missing: {dataset_path}", flush=True)
            continue

        context_length = cfg["context_length"]

        dataset_model_dir = os.path.join(trained_models_root, dataset_name)
        os.makedirs(dataset_model_dir, exist_ok=True)

        save_model_path = os.path.join(
            dataset_model_dir,
            f"model_{experiment_name}.pt"
        )

        if is_model_complete(save_model_path):
            skipped += 1
            continue

        command = [
            "python", "main.py",
            "--mode", "train",
            "--data", dataset_path,
            "--context_length", str(context_length),
            "--horizon", str(horizon),
            "--top_k", str(top_k),
            "--use_noise", "true" if use_noise else "false",
            "--norm", norm,
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--save_path", save_model_path,
            "--device", device,
        ]

        jobs.append((
            command,
            dataset_name,
            horizon,
            context_length,
            top_k,
            norm,
            dataset_path,
            device
        ))

print(f"\n{'='*60}")
print(f"  Jobs já completos (pulados) : {skipped}")
print(f"  Jobs a executar             : {len(jobs)}")
print(f"{'='*60}\n")

# ======================================
# RUN IN PARALLEL
# ======================================
Parallel(
    n_jobs=N_CORES,
    backend="loky",
    verbose=10
)(
    delayed(run_training)(
        command,
        dataset_name,
        horizon,
        context_length,
        top_k,
        norm,
        dataset_path,
        device
    )
    for (
        command,
        dataset_name,
        horizon,
        context_length,
        top_k,
        norm,
        dataset_path,
        device
    ) in jobs
)

print("All trainings completed.")

# ======================================
# OPTIONAL: RUN ANALYSIS PER EXPERIMENT
# ======================================
def run_experiment(top_k, norm, use_noise, epochs, lr):
    print(f"[INICIANDO] top_k={top_k} | norm={norm} | use_noise={use_noise} | epochs={epochs} | lr={lr}")

    experiment_name = build_experiment_name(top_k, norm, use_noise, epochs, lr)

    trained_models_root = os.path.join("trained_models_benchmark", experiment_name)

    run_full_experiment_pipeline(
        path_trained_models=trained_models_root,
        experiment_name="model_" + experiment_name,
        top_k=top_k,
        use_noise=use_noise
    )

Parallel(n_jobs=3)(
    delayed(run_experiment)(top_k, norm, use_noise, epochs, lr)
    for top_k, norm, use_noise, epochs, lr in EXPERIMENTS
)
