import subprocess
import os
import re
from itertools import product
from joblib import Parallel, delayed
from run_experiments_encoder import run_full_experiment_pipeline

# ======================================
# GENERAL CONFIGURATION
# ======================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

N_CORES = 20

base_path = "../all_datasets_global_by_years"
HORIZONS = [3, 6, 12, 24]

MAX_YEAR = 2024
MIN_YEAR = 2020

debug_state = None  # None
device = "cuda"

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
    """
    Retorna True se o arquivo .pt específico deste job já existe.
    Isso garante que apenas este job individual seja pulado,
    sem afetar os demais jobs do mesmo experimento.
    """
    return os.path.isfile(save_model_path)


def run_training(
    command,
    excluded_state,
    year,
    horizon,
    top_k,
    norm,
    dataset_path,
    device,
):
    print(
        f"Training: excluding={excluded_state} | "
        f"year={year} | horizon={horizon} | "
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
skipped = 0

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
        "trained_models_encoder",
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

                dataset_path = os.path.join(folder_path, dataset_file)

                save_model_path = os.path.join(
                    state_model_dir,
                    f"model_{experiment_name}_{year}.pt"
                )

                if is_model_complete(save_model_path):
                    skipped += 1
                    continue
                # ──────────────────────────────────────────────────────

                command = [
                    "python", "main_encoder.py",
                    "--mode", "train",
                    "--data", dataset_path,
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
        excluded_state,
        year,
        horizon,
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

    experiment_name = build_experiment_name(
        top_k, norm, use_noise, epochs, lr
    )

    trained_models_root = os.path.join(
        "trained_models_encoder",
        experiment_name
    )

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