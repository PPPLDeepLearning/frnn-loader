# -*- coding: utf-8 -*-

"""Dataset that combines multiple shots."""

import logging

from frnn_loader.loaders.frnn_dataset_disk import shot_dataset_disk


class frnn_multi_dataset():
    """Dataset that combines multiple shots."""
    def __init__(self, ds_list):
        """Instantiate multi dataset

        ds_list: list[shot_dataset_disk] - List of datasets to instantiate with
        """

        self.ds_list = ds_list

    def __getitem__(self, idx_slice):
        """Fetches a slice from a given dataset.
        
        Args:
            idx_slice: Tuple(int, slice) - Used to fetch a slice of data
        """
        print(f"frnn_multi_dataset.__getitem__: got ", idx_slice)
        ds_idx, slice = idx_slice
        return self.ds_list[ds_idx][slice]

# End of file frnn_multi_dataset.py
