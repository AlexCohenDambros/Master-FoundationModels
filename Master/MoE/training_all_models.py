import subprocess
import os
import re
from itertools import product
from joblib import Parallel, delayed
from run_experiments import run_full_experiment_pipeline

# ======================================
# GENERAL CONFIGURATION
# ======================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

N_CORES = 1

base_path = "../all_datasets_global_by_years"
HORIZONS = [3, 6, 12, 24]

BASE_CONTEXT = 410
MAX_YEAR = 2024
MIN_YEAR = 2020

debug_state = "sp"  # None 
device = "cpu"

# ======================================
# HYPERPARAMETER GRID
# ======================================
TOP_K_LIST = [2, 1]
NORM_LIST = ["std", "minmax"]
USE_NOISE_LIST = [True, False]
EPOCHS_LIST = [20, 30, 50, 100]
LR_LIST = [1e-4, 1e-3, 1e-5]

# ======================================
# HELPERS
# ======================================
def get_context_length(year: int, horizon: int) -> int:
    return BASE_CONTEXT - horizon - ((MAX_YEAR - year) * horizon)


def build_experiment_name(top_k, norm, use_noise, epochs, lr):
    return (
        f"topk_{top_k}"
        f"_norm_{norm}"
        f"_noise_{use_noise}"
        f"_ep_{epochs}"
        f"_lr_{lr}"
    )


def run_training(
    command,
    excluded_state,
    year,
    horizon,
    context_length,
    top_k,
    norm,
    dataset_path,
    device,
):
    print(f"Training: excluding={excluded_state} | " f"year={year} | horizon={horizon} | " f"context={context_length} | " f"top_k={top_k} | " f"norm={norm} | " f"use_noise={use_noise} | " f"epochs={epochs} | " f"lr={lr} | " f"data={dataset_path} | " f"device={device}", flush=True)

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

        print(
            f"✔ DONE | excluding={excluded_state} | year={year} | horizon={horizon}",
            flush=True
        )

        if result.stdout:
            print(result.stdout, flush=True)

        if result.stderr:
            print(result.stderr, flush=True)

    except subprocess.CalledProcessError as e:
        print(
            f"✖ FAILED | excluding={excluded_state} | year={year} | horizon={horizon}",
            flush=True
        )
        print(e.stderr, flush=True)


# ======================================
# BUILD JOB LIST (GRID SEARCH)
# ======================================
jobs = []

EXPERIMENTS = list(product(
    TOP_K_LIST,
    NORM_LIST,
    USE_NOISE_LIST,
    EPOCHS_LIST,
    LR_LIST
))

for top_k, norm, use_noise, epochs, lr in EXPERIMENTS:

    experiment_name = build_experiment_name(
        top_k, norm, use_noise, epochs, lr
    )

    trained_models_root = os.path.join(
        "trained_models",
        experiment_name
    )
    os.makedirs(trained_models_root, exist_ok=True)

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

            horizon_dir = os.path.join(trained_models_root, f"horizon_{HORIZON}")
            state_model_dir = os.path.join(horizon_dir, excluded_state_folder)
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

                context_length = get_context_length(year, HORIZON)
                if context_length <= 0:
                    continue

                dataset_path = os.path.join(folder_path, dataset_file)

                save_model_path = os.path.join(
                    state_model_dir,
                    f"model_{experiment_name}_{year}.pt"
                )

                if os.path.exists(save_model_path):
                    continue

                command = [
                    "python", "main.py",
                    "--mode", "train",
                    "--data", dataset_path,
                    "--context_length", str(context_length),
                    "--horizon", str(HORIZON),
                    "--top_k", str(top_k),
                    "--use_noise", "true" if use_noise else "false",
                    "--norm", norm,
                    "--epochs", str(epochs),
                    "--lr", str(lr),
                    "--save_path", save_model_path,
                    "--device", device
                ]

                jobs.append((
                    command,
                    excluded_state,
                    year,
                    HORIZON,
                    context_length,
                    top_k,
                    norm,
                    dataset_path,
                    device
                ))

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
        excluded_state,
        year,
        horizon,
        context_length,
        top_k,
        norm,
        dataset_path,
        device
    )
    for (
        command,
        excluded_state,
        year,
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
for top_k, norm, use_noise, epochs, lr in EXPERIMENTS:
    experiment_name = build_experiment_name(
        top_k, norm, use_noise, epochs, lr
    )

    trained_models_root = os.path.join(
        "trained_models",
        experiment_name
    )

    run_full_experiment_pipeline(
        path_trained_models=trained_models_root,
        experiment_name="model_" + experiment_name,
        top_k=top_k,
        use_noise=use_noise
    )
