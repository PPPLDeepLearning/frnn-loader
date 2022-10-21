#-*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_dataloader.py
import unittest
import tempfile
import logging
import shutil
import errno
from os import environ

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from frnn_loader.primitives.resamplers import resampler_causal
from frnn_loader.backends.backend_hdf5 import backend_hdf5
from frnn_loader.backends.backend_dummy import backend_dummy
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.loaders.frnn_dataset_disk import shot_dataset_disk
from frnn_loader.loaders.frnn_multi_dataset import frnn_multi_dataset
from frnn_loader.loaders.frnn_loader import random_sequence_sampler, batched_random_sequence_sampler

FORMAT = "%(asctime)s unittest test_frnndataset %(message)s"
logging.basicConfig(format=FORMAT,level=logging.DEBUG)


class test_frnn_loader(unittest.TestCase):
    """Test FRNN dataloader."""

    @classmethod
    def setUpClass(cls):
        """Set up unit tests:

        * Create a temporary directory
        """
        try:
            cls.root = tempfile.mkdtemp(dir=environ["TMPDIR"])
        except KeyError:
            cls.root = tempfile.mkdtemp(dir="/home/rkube/tmp")

        cls.shotlist = [180619, 180620]
        cls.signal_list = ["dssdenest", "q95", "echpwrc", "pradcore", "ipsiptargt"]

        # Instantiate resampler, etc.  
        cls.my_resampler = resampler_causal(0.0, 2e3, 1e0)
        # Instantiate a file backend
        cls.my_backend_file = backend_hdf5(cls.root)
        cls.my_fetcher = fetcher_d3d_v1()
        cls.pred_list = [signal_0d(n) for n in cls.signal_list]

        cls.ds_list = [shot_dataset_disk(shotnr, 
                                         predictors=cls.pred_list, 
                                         resampler=cls.my_resampler,
                                         backend_file=cls.my_backend_file, 
                                         fetcher=cls.my_fetcher, 
                                         root = cls.root,
                                         download=True,
                                         dtype=torch.float32) for shotnr in cls.shotlist]
        cls.ds_multi = frnn_multi_dataset(cls.ds_list)

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

    def test_frnn_loader_single(self):
        """Test instantiation and iteration over multi-shot dataset."""

        my_sampler = random_sequence_sampler(self.ds_list, seq_length=222)
        my_loader = DataLoader(self.ds_multi, sampler=my_sampler)

        # test no-batching
        num_batches = 0
        for v in my_loader:
            print(v[0].shape, v[1].shape)
            num_batches += 1
        print(f"num_batches = {num_batches}")


    # def test_frnn_loader_batched(self):
    #     """Test instantiation and iteration over multi-shot dataset."""
    #     def batch_collate_fn(input):
    #         # Somehow the input has been wrapped in a list. 
    #         # No idea where this comes from
    #         return torch.stack(input[0])
    #     batch_size = 3

    #     my_sampler = batched_random_sequence_sampler(self.ds_list, seq_length=55, batch_size=batch_size, drop_last=False)
    #     my_loader = DataLoader(self.ds_multi, sampler=my_sampler, collate_fn=batch_collate_fn)


    #     plt.figure()
    #     # test no-batching
    #     num_batches = 0
    #     for v in my_loader:
    #         print(v.shape)
    #         num_batches += 1

    #         for b in range(batch_size):
    #             plt.plot(v[b, :, 2])
    #     print(f"batch_size = {batch_size}: num_batches = {num_batches}")

    #     plt.show()