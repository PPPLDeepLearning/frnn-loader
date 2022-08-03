# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_backends.py
import unittest

import torch
from frnn_loader.backends.machine import MachineD3D
from frnn_loader.data.user_signals import fs07, ip, q95, neped


from frnn_loader.primitives.resamplers import resampler_last
from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.loaders.frnn_dataset import shot_dataset


class test_frnn_dataset(unittest.TestCase):
    """Test routines working with the frnn dataset."""
    def test_frnn_dataset(self):
        """Test instantiation of the dataset."""
        # Instantiate a resampler
        my_resampler = resampler_last(0.0, 2.0, 1e-3)

        # Instantiate a file backend
        my_backend_file = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021/")

        ds = shot_dataset(184800, MachineD3D(), [fs07, q95, neped], resampler=my_resampler, backend_file=my_backend_file, download=False, dtype=torch.float32)
        print(ds)


if __name__ == "__main__":
    unittest.main()

# end of file test_frnndataset.py
