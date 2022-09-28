# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_dataset_disk.py
import unittest
import tempfile
import logging
import shutil
import errno
from os import environ

import torch
import numpy as np

from frnn_loader.primitives.resamplers import resampler_causal
from frnn_loader.backends.backend_hdf5 import backend_hdf5
from frnn_loader.backends.backend_dummy import backend_dummy
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
        try:
            cls.root = tempfile.mkdtemp(environ["TMPDIR"])
        except KeyError:
            cls.root = tempfile.mkdtemp(dir="/home/rkube/tmp/")
    
        cls.shotnr = 180619
        cls.signal_list = ["dssdenest", "fs07", "q95", "qmin", "efsli", "ipspr15V", "efsbetan",
                     "efswmhd", "dusbradial", "echpwrc", "pradcore", "pradedge", "bmspinj", "bmstinj",
                     "iptdirect", "ipsiptargt", "ipeecoil",
                     "tmamp1", "tmamp2", "tmfreq1", "tmfreq2"]

    @classmethod
    def tearDownClass(cls):
        """Tear down unit backend tests.
        
        * Delete temp directory.
        """
        try:
            shutil.rmtree(cls.root)  # delete directory
        except OSError as exc:
            if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                raise  # re-raise exception

    def test_frnn_dataset(self):
        """Test instantiation of the dataset."""
        # Instantiate a resampler
        my_resampler = resampler_causal(0.0, 2e3, 1e0)

        # Instantiate a file backend
        my_backend_file = backend_hdf5(self.root)
        my_fetcher = fetcher_d3d_v1()
        pred_list = [signal_0d(n) for n in self.signal_list]

        ds = shot_dataset_disk(self.shotnr, 
                               predictors=pred_list, 
                               resampler=my_resampler, 
                               backend_file=my_backend_file, 
                               fetcher=my_fetcher, 
                               root = self.root,
                               download=True,
                               dtype=torch.float32)

    def test_dataset_slicing(self):
        """Test slicing operations on dataset."""

        my_resampler = resampler_causal(0.0, 1e3, 1.0)
        sig_dummy1 = signal_0d("dummy1", sig_def_fname="dummy_signals.yaml")
        sig_dummy2 = signal_0d("dummy2", sig_def_fname="dummy_signals.yaml")
        my_backend_dummy = backend_dummy(None)

        # Instantiate dummy dataset.
        # The dummy data is just torch.arange(0.0, 1000.0, 1.0) in each signal
        ds_dummy = shot_dataset_disk(1, predictors=[sig_dummy1, sig_dummy2], 
                                     resampler=my_resampler, 
                                     backend_file=my_backend_dummy,
                                     fetcher=None, 
                                     root=self.root, 
                                     download=False, dtype=torch.float32)

        # Now we test that the index, passed to the dataset __getitem__ method
        # are roughly the the same as we expect from a torch.tensor.
        #
        # For comparison, the data we are asccessing is
        data_comp = torch.stack([torch.arange(0.0, 1000.0, 1.0),
                                 torch.arange(0.0, 1000.0, 1.0)]).T

        # Test individual element access
        for idx in np.random.randint(0, 1000, 100):
            assert((ds_dummy[idx] == data_comp[idx]).all())

        # Test slicing
        assert((ds_dummy[:] == data_comp[:]).all())
        assert((ds_dummy[1:] == data_comp[1:]).all())
        assert((ds_dummy[3:-9] == data_comp[3:-9]).all())
        assert((ds_dummy[1:44:2] == data_comp[1:44:2]).all())
        
        # Test indexing using index tuples / listst
        idx = torch.randperm(1000)[:100]
        assert((ds_dummy[idx] == data_comp[idx]).all())

        idx = list(torch.randperm(1000)[:100])
        assert((ds_dummy[idx] == data_comp[idx]).all())


if __name__ == "__main__":
    unittest.main()
