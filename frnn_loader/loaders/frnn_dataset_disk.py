# -*- coding: utf-8 -*-

"""FRNN Dataset """

import torch
from torch.utils.data import Dataset
import logging

from frnn_loader.primitives.signal import signal_0d

class shot_dataset_disk(Dataset):
    """Dataset representing a shot which uses disk data as storage
    
    This dataset maps the __getitem__ method to a pre-cached version on disk.
    The cached version on disk is built 
    * Using the predictors and targets 
    * With all transformations applied to them.
    
    """
    def __init__(self, shotnr, predictors, resampler, backend_file, fetcher, cache, download=False, transform=None, dtype=torch.float32):
        """Initializes the disk dataset."""
        self.shotnr = shotnr
        self.predictors = predictors
        self.resampler = resampler
        self.backend_file = backend_file
        self.fetcher = fetcher
        self.cache = cache
        self.download = download
        self.transform = transform
        self.dtype = dtype

        # If we want to download we need to have a fetcher bassed
        if self.download:
            assert self.fetcher is not None

        # Next is pre-processing. We attack it like this
        # 1. Fetch the signals
        # 2. Apply the transformation
        # 3. Store the data in a hdf5 file.







# end of file frnn_dataset_disk.py
