import subprocess
import os
import re

# ======================================
# GENERAL CONFIGURATION
# ======================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Base dataset path (root)
base_path = "../all_datasets_global_by_years"

# Horizons to train
HORIZONS = [3, 6, 12, 24]

# Base context starting point
BASE_CONTEXT = 410
MAX_YEAR = 2024
MIN_YEAR = 2020

# Root directory for trained models
trained_models_root = "trained_models"
os.makedirs(trained_models_root, exist_ok=True)

# ======================================
# HELPER: Dynamic context length
# ======================================
def get_context_length(year: int, horizon: int) -> int:
    return BASE_CONTEXT - horizon - ((MAX_YEAR - year) * horizon)

# ======================================
# MAIN LOOP
# ======================================
for HORIZON in HORIZONS:
    print(f"\n=== Processing horizon {HORIZON} ===")

    # Base path for this horizon
    horizon_base_path = os.path.join(base_path, f"horizon_{HORIZON}")
    if not os.path.exists(horizon_base_path):
        print(f"Path not found: {horizon_base_path}")
        continue

    # Loop through excluded state folders
    for excluded_state_folder in os.listdir(horizon_base_path):

        # expecting: excluding_STATE
        if not excluded_state_folder.startswith("excluding_"):
            continue

        folder_path = os.path.join(horizon_base_path, excluded_state_folder)
        if not os.path.isdir(folder_path):
            continue

        # Extract state name
        excluded_state = excluded_state_folder.replace("excluding_", "")

        # Format: trained_models/horizon_X/excluding_STATE/
        horizon_dir = os.path.join(trained_models_root, f"horizon_{HORIZON}")
        state_model_dir = os.path.join(
            horizon_dir, excluded_state_folder
        )
        os.makedirs(state_model_dir, exist_ok=True)

        # Loop through yearly datasets
        for dataset_file in os.listdir(folder_path):
            if not dataset_file.endswith(".jsonl"):
                continue

            match = re.search(r"dataset_(\d{4})\.jsonl", dataset_file)
            if not match:
                print(f"Could not extract year from file: {dataset_file}")
                continue

            year = int(match.group(1))

            # Restrict years
            if year < MIN_YEAR or year > MAX_YEAR:
                continue

            dataset_path = os.path.join(folder_path, dataset_file)

            # Dynamic context length
            context_length = get_context_length(year, HORIZON)

            if context_length <= 0:
                print(
                    f"Invalid context length ({context_length}) "
                    f"for {year} (horizon {HORIZON}). Skipping."
                )
                continue

            # Save path
            save_model_path = os.path.join(
                state_model_dir,
                f"model_excluding_{excluded_state}_{year}.pt"
            )

            # Command
            command = [
                "python", "main.py",
                "--mode", "train",
                "--data", dataset_path,
                "--context_length", str(context_length),
                "--horizon", str(HORIZON),
                "--save_path", save_model_path,
                "--device", "cpu"
            ]

            print(command)

            print(
                f"Training: excluding={excluded_state} | "
                f"year={year} | horizon={HORIZON} | "
                f"context={context_length} | "
                f"data={dataset_path} | "
            )

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
                print(f"Training failed: {dataset_file}")
                print(e.stderr)

print("All trainings completed.")
