# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_backends.py
import unittest
from xml.sax.handler import feature_external_ges
import torch

import logging

from frnn_loader.primitives.resamplers import resampler_last
from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.loaders.frnn_dataset import shot_dataset

FORMAT = "%(asctime)s unittest test_frnndataset %(message)s"
logging.basicConfig(format=FORMAT)


class test_frnn_dataset(unittest.TestCase):
    """Test routines working with the frnn dataset."""

    def test_frnn_dataset(self):
        """Test instantiation of the dataset."""
        # Instantiate a resampler
        my_resampler = resampler_last(0.0, 2.0, 1e-3)

        # Instantiate a file backend
        my_backend_file = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021/")
        my_fetcher = fetcher_d3d_v1()

        signal_fs07 = signal_0d("fs07")
        signal_q95 = signal_0d("q95")


        ds = shot_dataset(184800, [signal_fs07, signal_q95], 
                          resampler=my_resampler, backend_file=my_backend_file, 
                          backend_fetcher=my_fetcher, download=True,
                          dtype=torch.float32)
        print(ds.signal_tensor.shape)
        #for item in ds:
        #    print(item.shape)


if __name__ == "__main__":
    unittest.main()

# end of file test_frnndataset.py
