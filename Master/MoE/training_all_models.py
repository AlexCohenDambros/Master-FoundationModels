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

# Base path of the previously created datasets
base_path = "../all_datasets_global_by_years"

# Fixed horizon
HORIZON = 12

# Mapping from year to context_length
context_by_year = {
    2024: 398,
    2023: 386,
    2022: 374,
    2021: 362,
    2020: 350
}

# Root directory where all trained models will be stored
trained_models_root = "trained_models"

# Create root directory if it doesn't exist
os.makedirs(trained_models_root, exist_ok=True)

# ======================================
# LOOP THROUGH ALL DATASETS
# ======================================
for excluded_state_folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, excluded_state_folder)

    if not os.path.isdir(folder_path):
        continue  # Skip non-directory files

    # Create a subfolder for the current state inside trained_models
    state_model_dir = os.path.join(trained_models_root, excluded_state_folder)
    os.makedirs(state_model_dir, exist_ok=True)

    # Loop through all datasets by year
    for dataset_file in os.listdir(folder_path):
        if dataset_file.endswith(".jsonl"):
            dataset_path = os.path.join(folder_path, dataset_file)

            # Extract year from filename (e.g., dataset_2024.jsonl)
            match = re.search(r"dataset_(\d{4})\.jsonl", dataset_file)
            if not match:
                print(f"Could not extract year from file: {dataset_file}")
                continue

            year = int(match.group(1))

            # Determine context_length
            if year in context_by_year:
                context_length = context_by_year[year]
            else:
                print(f"Year {year} is not in the mapping. Skipping file {dataset_file}")
                continue

            # Model save file path inside the state's folder
            save_model_path = os.path.join(
                state_model_dir, f"model_{excluded_state_folder}_{year}.pt"
            )

            # ======================================
            # RUN TRAINING
            # ======================================
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
                print(f"Training failed for {dataset_file} with exit code {e.returncode}")
                print(e.stderr)

print("All trainings have been completed.")