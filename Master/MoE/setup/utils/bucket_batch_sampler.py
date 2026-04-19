import random
from torch.utils.data import Sampler
from collections import defaultdict
import torch
from .time_series_dataset import TimeSeriesDataset

# -------------------
# Bucket Batch Sampler
# -------------------
class BucketBatchSampler(Sampler):
    """
    Groups series by exact context length so all samples in a batch share the
    same length. This allows torch.stack in the collate function and enables
    fully vectorized expert calls — no padding or per-sample loops needed.

    Usage with DataLoader:
        sampler = BucketBatchSampler(dataset, batch_size=32)
        loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    """

    def __init__(self, dataset: TimeSeriesDataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by context length — stored so __iter__ can rebuild each epoch
        self.length_to_indices: dict = defaultdict(list)
        for idx, (inp, _) in enumerate(dataset.samples):
            self.length_to_indices[len(inp)].append(idx)

        # Build once for __len__
        self.batches = self._build_batches()

    def _build_batches(self) -> list:
        """
        Rebuilds the batch list from scratch.
        When shuffle=True, re-shuffles indices within each bucket (new composition)
        and then re-shuffles batch order — called every epoch via __iter__.
        """
        batches = []
        for indices in self.length_to_indices.values():
            indices = list(indices)          # copy — never mutate the stored groups
            if self.shuffle:
                random.shuffle(indices)      # new composition each epoch
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i : i + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)          # new order each epoch
        return batches

    def __iter__(self):
        # Rebuild every epoch when shuffle=True so composition AND order both change
        batches = self._build_batches() if self.shuffle else self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)

def collate_fn(batch):
    """
    Stacks a batch of (inp, tgt) tensors.
    BucketBatchSampler guarantees all inp tensors in a batch have the same
    length, so torch.stack works without padding.
    """
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)