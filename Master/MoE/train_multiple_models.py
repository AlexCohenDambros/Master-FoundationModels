import subprocess
import os
from pathlib import Path

# =====================
# Environment settings
# =====================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# =====================
# Directories
# =====================
datasets_dir = Path("../all_datasets_global")       # folder containing .jsonl files
results_dir = Path("trained_models")                    # output folder
results_dir.mkdir(exist_ok=True)                 # create it if it doesn't exist

# List all .jsonl files in the datasets directory
dataset_files = sorted(datasets_dir.glob("*.jsonl"))

print(f"Found {len(dataset_files)} dataset files to process.")

# =====================
# Loop over datasets
# =====================
for dataset_path in dataset_files:
    dataset_name = dataset_path.stem                  

    short_name = dataset_name.split("_", 1)[-1]
    save_path = results_dir / f"model_excluding_{short_name}.pt"

    print(f"\nProcessing dataset: {dataset_name}")

    command = [
        "python", "main.py",
        "--mode", "train",
        "--data", str(dataset_path),
        "--context_length", "398",
        "--horizon", "12",
        "--save_path", str(save_path)
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Failed to process {dataset_name} (exit code {e.returncode})")
        print(e.stderr)
