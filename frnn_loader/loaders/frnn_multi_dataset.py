# -*- coding: utf-8 -*-

"""Dataset that combines multiple shots."""

import logging
import torch
from frnn_loader.loaders.frnn_dataset_disk import shot_dataset_disk


class frnn_multi_dataset:
    """Dataset that combines multiple shots."""

    def __init__(self, ds_list):
        """Instantiate multi dataset

        ds_list: list[shot_dataset_disk] - List of datasets to instantiate with
        """

        self.ds_list = ds_list

    def __getitem__(self, idx_slice):
        """Fetches a slice from a given dataset.

        Data is addressed using an idx_slice, which is tuple (i, slice)
        where i indeces the dataset(shot) and slice defines the data whithin shot i.

        For batching, it is convenient to receive a list of these.
        This is implemented f.ex. in ~frnn_loader.loaders.frnn_loader.random_sequence_sampler

        Args:
            idx_slice: int - Used to fetch a single dataset
            idx_slice: Tuple(int, slice) - Used to fetch a slice of data
            idx_slice: List(Tuple(int, slice)) - Used for batching.
        """
        if isinstance(idx_slice, int):
            # Just fetch a single dataset
            return self.ds_list[idx_slice]
        elif isinstance(idx_slice, tuple):
            # If idx_slice is a tuple, map the slice directly to the requested dataset
            ds_idx, slice = idx_slice
            return self.ds_list[ds_idx][slice]
        elif isinstance(idx_slice, list):
            # If idx_slice is a list, do the mapping of index and slice individually
            # and return a list.

            return [self.ds_list[ds_idx][slice] for ds_idx, slice in idx_slice]

    def __iter__(self):
        """Iterate over dataset list."""
        if not self.ds_list:
            raise StopIteration
        yield from self.ds_list


# End of file frnn_multi_dataset.py
