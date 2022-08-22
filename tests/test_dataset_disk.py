# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_dataset_disk.py
import unittest
import tempfile
import logging

import torch

from frnn_loader.primitives.resamplers import resampler_causal
from frnn_loader.backends.backend_hdf5 import backend_hdf5
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.loaders.frnn_dataset_disk import shot_dataset_disk

FORMAT = "%(asctime)s unittest test_frnndataset %(message)s"
logging.basicConfig(format=FORMAT,level=logging.DEBUG)


class test_disk_dataset(unittest.TestCase):
    """Test routines wirking with the disk dataset."""

    @classmethod
    def setUpClass(cls):
        """Set up unit tests for disk dataset.
        
        * Create a temporary directory
        """
        cls.root = tempfile.mkdtemp(dir="/home/rkube/tmp")
        cls.shotnr = 180619
        cls.signal_list = ["fs07", "q95"]

    @classmethod
    def tearDownClass(cls):
        """Tear down unit backend tests.
        
        * Delete temp directory.
        """
        # try:
        #     shutil.rmtree(cls.root)  # delete directory
        # except OSError as exc:
        #     if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
        #         raise  # re-raise exception

    def test_frnn_dataset(self):
        """Test instantiation of the dataset."""
        # Instantiate a resampler
        my_resampler = resampler_causal(0.0, 2e3, 1e0)

        # Instantiate a file backend
        my_backend_file = backend_hdf5("/home/rkube/datasets/frnn/")
        my_fetcher = fetcher_d3d_v1()
        root = self.root

        signal_fs07 = signal_0d("fs07")
        signal_q95 = signal_0d("q95")

        ds = shot_dataset_disk(self.shotnr, 
                               predictors=[signal_fs07, signal_q95], 
                               resampler=my_resampler, 
                               backend_file=my_backend_file, 
                               fetcher=my_fetcher, 
                               root = root,
                               download=True,
                               dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
