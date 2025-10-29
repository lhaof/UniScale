from typing import Sequence

from torch.utils.data import BatchSampler, Sampler

from mmdet.registry import DATA_SAMPLERS
import random
import numpy as np

@DATA_SAMPLERS.register_module()
class AspectRatioBatchSampler_SameDataset(BatchSampler):
    """
    Batch sampler that:
    1. Ensures each batch contains samples from the same dataset.
    2. Groups images with similar aspect ratio (<1 or >=1).
    3. Allows balanced or weighted sampling across multiple datasets.
    4. Automatically resets and reshuffles each dataset when exhausted.
    """
    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 dataset_weights: list = None,
                 drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(f'sampler should be an instance of Sampler, but got {type(sampler)}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f'batch_size should be a positive integer, but got {batch_size}')

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.dataset = self.sampler.dataset
        self.datasets = self.dataset._metainfo['datasets']
        self.num_datasets = len(self.datasets)

        # split indices by dataset
        self.dataset_indices = [[] for _ in range(self.num_datasets)]
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            data_id = data_info['dataset_id']
            self.dataset_indices[data_id].append(idx)

        # shuffle each dataset index list
        for lst in self.dataset_indices:
            random.shuffle(lst)

        # import pdb; pdb.set_trace()
        # dataset_iters: [num_datasets]
        self.dataset_iters = [iter(lst) for lst in self.dataset_indices]

        # aspect ratio buckets: [num_datasets][2]
        self._aspect_dataset_buckets = [[[] for _ in range(2)] for _ in range(self.num_datasets)]

        # normalize weights
        if dataset_weights is None:
            dataset_weights = [1.0] * self.num_datasets
        total = sum(dataset_weights)
        self.dataset_weights = [w / total for w in dataset_weights]

    def _reset_iterator(self, dataset_id: int):
        """Reshuffle and reset the iterator for a specific dataset."""
        random.shuffle(self.dataset_indices[dataset_id])
        self.dataset_iters[dataset_id] = iter(self.dataset_indices[dataset_id])

    def __iter__(self) -> Sequence[int]:
        while True:
            # pick a dataset based on weights
            # import pdb; pdb.set_trace()
            dataset_id = random.choices(range(self.num_datasets), weights=self.dataset_weights, k=1)[0]

            # try to fetch next sample
            try:
                idx = next(self.dataset_iters[dataset_id])
            except StopIteration:
                # reset that dataset and retry
                self._reset_iterator(dataset_id)
                idx = next(self.dataset_iters[dataset_id])

            data_info = self.dataset.get_data_info(idx)
            width, height = data_info['width'], data_info['height']
            bucket_id = 0 if width < height else 1
            bucket = self._aspect_dataset_buckets[dataset_id][bucket_id]
            bucket.append(idx)

            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]


    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size