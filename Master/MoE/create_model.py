import subprocess
import os 

os.environ["TF_ENABLE_ONEDNN_OPTS"] ="0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

command = [
    "python", "main.py",
    "--mode", "train",
    "--data", "../dataset_global_nosp/dataset_global_nosp.jsonl",
    "--context_length", "398",
    "--horizon", "12",
    "--save_path", "sem_valid.pt"
]

try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

except subprocess.CalledProcessError as e:
    print(f"Command failed with exit code {e.returncode}")
    print(e.stderr)
