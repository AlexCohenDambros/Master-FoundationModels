import torch 
from torch.utils.data import Dataset
import json

# -------------------
# Dataset
# -------------------
class TimeSeriesDataset(Dataset):
    """
    Variable-length time series dataset.

    For each sequence of length L:
        inp = seq[:-horizon]   (variable length: L - horizon)
        tgt = seq[-horizon:]   (fixed length: horizon)

    This removes the fixed context_length constraint, allowing a single model
    to be trained and applied across datasets with different series lengths.
    """

    def __init__(self, sequences, horizon: int):
        self.samples = []
        for seq in sequences:
            if len(seq) > horizon:
                inp = seq[:-horizon]
                tgt = seq[-horizon:]
                self.samples.append((inp, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return (
            torch.tensor(inp, dtype=torch.float32),
            torch.tensor(tgt, dtype=torch.float32),
        )

# -------------------
# Load Data
# -------------------
def load_jsonl(path: str):
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seq = obj.get("sequence")
            if seq is None:
                raise ValueError("Each JSONL line must have the key 'sequence'")
            seqs.append([float(x) for x in seq])
    return seqs