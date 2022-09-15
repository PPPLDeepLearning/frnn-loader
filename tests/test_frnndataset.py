# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_frnndataset.py
import unittest
import torch
import shutil
import tempfile
import errno
from os import environ


import logging

from frnn_loader.primitives.resamplers import resampler_causal
from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.loaders.frnn_dataset import shot_dataset

FORMAT = "%(asctime)s unittest test_frnndataset %(message)s"
logging.basicConfig(format=FORMAT,level=logging.DEBUG)


class test_frnn_dataset(unittest.TestCase):
    """Test routines working with the frnn dataset."""

    @classmethod
    def setUpClass(cls):
        """Set up unit tests for disk dataset.
        
        * Create a temporary directory
        """
        try:
            cls.root = environ["TMPDIR"]
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
        my_resampler = resampler_causal(0.0, 2.0, 1e-3)

        # Instantiate a file backend
        my_backend_file = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021/")
        my_fetcher = fetcher_d3d_v1()


        signal_names = ["dssdenest", "fs07", "q95", "qmin", "efsli", "ipspr15V", "efsbetan",
                        "efswmhd", "dusbradial", "echpwrc", "pradcore", "pradedge", "bmspinj", 
                        "bmstinj", "iptdirect", "ipsiptargt", "ipeecoil",
                        "tmamp1", "tmamp2", "tmfreq1", "tmfreq2"]

        pred_list = [signal_0d(n) for n in signal_names]

        ds = shot_dataset(184800, pred_list,
                          resampler=my_resampler, backend_file=my_backend_file, 
                          backend_fetcher=my_fetcher, download=True,
                          dtype=torch.float32)
        print(ds.signal_tensor.shape)
        #for item in ds:
        #    print(item.shape)


if __name__ == "__main__":
    unittest.main()

# end of file test_frnndataset.py
