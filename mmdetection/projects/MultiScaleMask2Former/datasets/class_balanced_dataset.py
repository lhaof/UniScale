import bisect
import copy
import logging
import math
from collections import defaultdict
from typing import List, Sequence, Tuple, Union

import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from mmengine.logging import print_log
from mmengine.registry import DATASETS
from mmengine.dataset.base_dataset import BaseDataset, force_full_init

@DATASETS.register_module()
class ClassBalancedDataset_MS:
    """A wrapper of class balanced dataset.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :meth:`get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Note:
        ``ClassBalancedDataset`` should not inherit from ``BaseDataset``
        since ``get_subset`` and ``get_subset_`` could  produce ambiguous
        meaning sub-dataset which conflicts with original dataset. If you
        want to use a sub-dataset of ``ClassBalancedDataset``, you should set
        ``indices`` arguments for wrapped dataset which inherit from
        ``BaseDataset``.

    Args:
        dataset (BaseDataset or dict): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 oversample_thr: float,
                 lazy_init: bool = False):
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        self.oversample_thr = oversample_thr
        self._metainfo = self.dataset.metainfo

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        # Get repeat factors for each image.
        repeat_factors = self._get_repeat_factors(self.dataset,
                                                  self.oversample_thr)
        # Repeat dataset's indices according to repeat_factors. For example,
        # if `repeat_factors = [1, 2, 3]`, and the `len(dataset) == 3`,
        # the repeated indices will be [1, 2, 2, 3, 3, 3].
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        self._fully_initialized = True


    def _get_repeat_factors(self, dataset: BaseDataset,
                            repeat_thr: List[float]) -> List[float]:
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (BaseDataset): The dataset.
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            List[float]: The repeat factors for each images in the dataset.
        """
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq: defaultdict = defaultdict(float)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            assert v > 0, f'caterogy {k} does not contain any images'
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr[cat_id] / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I and its labels L(I), compute the image-level
        # repeat factor:
        #    r(I) = max_{c in L(I)} r(c)
        repeat_factors = []
        for idx in range(num_images):
            # the length of `repeat_factors` need equal to the length of
            # dataset. Hence, if the `cat_ids` is empty,
            # the repeat_factor should be 1.
            repeat_factor: float = 1.
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) != 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            int: Local index of data.
        """
        return self.repeat_indices[idx]

    @force_full_init
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids of class balanced dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_cat_ids(sample_idx)


    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)


    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        ori_index = self._get_ori_dataset_idx(idx)
        return self.dataset[ori_index]

    @force_full_init
    def __len__(self):
        return len(self.repeat_indices)

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``ClassBalancedDataset`` for the ambiguous meaning
        of sub-dataset."""
        raise NotImplementedError(
            '`ClassBalancedDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ClassBalancedDataset`.')


    def get_subset(self, indices: Union[List[int], int]) -> 'BaseDataset':
        """Not supported in ``ClassBalancedDataset`` for the ambiguous meaning
        of sub-dataset."""
        raise NotImplementedError(
            '`ClassBalancedDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ClassBalancedDataset`.')