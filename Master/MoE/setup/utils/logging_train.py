from pathlib import Path
import csv

def get_log_dir_from_save_path(save_path):
    save_path = Path(save_path)
    log_dir = save_path.with_suffix("")  
    log_dir = log_dir.parent / f"{log_dir.name}_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def append_experts_weights(
    csv_path,
    sample_idx,
    learners,
    weights
):
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Sample", "Learner", "Weight"])

        for learner, weight in zip(learners, weights):
            writer.writerow([
                f"Sample {sample_idx}",
                learner,
                float(weight)
            ])

def append_train_loss(
    csv_path,
    epoch,
    train_loss
):
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Epoch", "Train Loss"])

        writer.writerow([
            int(epoch),
            float(train_loss)
        ])