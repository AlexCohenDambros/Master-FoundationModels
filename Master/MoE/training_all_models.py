import subprocess
import os
import re
from joblib import Parallel, delayed

N_CORES = 10

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

base_path = "../all_datasets_global_by_years"
HORIZONS = [12, 24]
BASE_CONTEXT = 410
MAX_YEAR = 2024
MIN_YEAR = 2020

trained_models_root = "trained_models"
os.makedirs(trained_models_root, exist_ok=True)

def get_context_length(year: int, horizon: int) -> int:
    return BASE_CONTEXT - horizon - ((MAX_YEAR - year) * horizon)

def run_training(command):
    print("Running:", " ".join(command))
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {command}")
        print(e.stderr)

tasks = []

for HORIZON in HORIZONS:
    horizon_base_path = os.path.join(base_path, f"horizon_{HORIZON}")
    if not os.path.exists(horizon_base_path):
        continue

    for excluded_state_folder in os.listdir(horizon_base_path):
        if not excluded_state_folder.startswith("excluding_"):
            continue

        folder_path = os.path.join(horizon_base_path, excluded_state_folder)
        if not os.path.isdir(folder_path):
            continue

        excluded_state = excluded_state_folder.replace("excluding_", "")
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
            context_length = get_context_length(year, HORIZON)
            if context_length <= 0:
                continue

            save_model_path = os.path.join(
                state_model_dir,
                f"model_excluding_{excluded_state}_{year}.pt"
            )

            command = [
                "python", "main.py",
                "--mode", "train",
                "--data", dataset_path,
                "--context_length", str(context_length),
                "--horizon", str(HORIZON),
                "--save_path", save_model_path,
                "--device", "cpu"
            ]

            tasks.append(command)

Parallel(n_jobs=N_CORES)(
    delayed(run_training)(cmd) for cmd in tasks
)

print("All trainings completed.")
