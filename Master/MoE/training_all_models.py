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

# Base dataset path
base_path = "../all_datasets_global_by_years"

# Fixed forecast horizon
HORIZON = 12

# Context length per year
context_by_year = {
    2024: 398,
    2023: 386,
    2022: 374,
    2021: 362,
    2020: 350
}

# Root directory for trained models
trained_models_root = "trained_models"
os.makedirs(trained_models_root, exist_ok=True)

# ======================================
# MAIN LOOP
# ======================================
for excluded_state_folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, excluded_state_folder)
    if not os.path.isdir(folder_path):
        continue

    state_model_dir = os.path.join(trained_models_root, excluded_state_folder)

    # Skip if this state's folder already exists
    if os.path.exists(state_model_dir):
        print(f"Folder '{state_model_dir}' already exists. Skipping.")
        continue

    os.makedirs(state_model_dir, exist_ok=False)

    # Loop through yearly datasets
    for dataset_file in os.listdir(folder_path):
        if not dataset_file.endswith(".jsonl"):
            continue

        dataset_path = os.path.join(folder_path, dataset_file)
        match = re.search(r"dataset_(\d{4})\.jsonl", dataset_file)
        if not match:
            print(f"Could not extract year from file: {dataset_file}")
            continue

        year = int(match.group(1))
        if year not in context_by_year:
            print(f"Year {year} not in context mapping. Skipping {dataset_file}.")
            continue

        context_length = context_by_year[year]
        save_model_path = os.path.join(
            state_model_dir, f"model_{excluded_state_folder}_{year}.pt"
        )

        # Run training
        command = [
            "python", "main.py",
            "--mode", "train",
            "--data", dataset_path,
            "--context_length", str(context_length),
            "--horizon", str(HORIZON),
            "--save_path", save_model_path
        ]

        print(f"Training model for {excluded_state_folder} - {year} "
              f"(context_length={context_length})...")

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Training failed for {dataset_file} (exit code {e.returncode})")
            print(e.stderr)

print("All trainings completed.")