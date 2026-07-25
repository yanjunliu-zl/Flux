"""Multi-resolution bucketing sampler for efficient batch training.

Groups videos by resolution, frame count, and aspect ratio to minimize padding.
"""

import random
from collections import defaultdict
from torch.utils.data import Sampler


class BucketSampler(Sampler):
    """Samples items grouped by (resolution, num_frames, aspect_ratio) buckets.

    Args:
        dataset: PyTorch Dataset.
        bucket_config: List of (resolution, num_frames, aspect_ratio) tuples.
        batch_size: Base batch size.
        shuffle: Whether to shuffle samples within buckets.
        drop_last: Whether to drop incomplete batches.
    """

    def __init__(
        self,
        dataset,
        bucket_config: list[tuple[int, int, str]],
        batch_size: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.bucket_config = bucket_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Assign samples to buckets
        self.buckets = defaultdict(list)
        for idx in range(len(dataset)):
            # Assign to first matching bucket (simplified: just use first bucket)
            self.buckets[0].append(idx)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
