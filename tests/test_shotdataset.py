# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_shotdataset.py

import unittest

import torch

from frnn_loader.backends.machine import MachineD3D
from frnn_loader.data.user_signals import fs07, ip, q95, neped
#from frnn_loader.primitives.shots import Shot
#from frnn_loader.utils.processing import resample_signal

from frnn_loader.primitives.resamplers import resampler_last
from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.loaders.frnn_dataset import shot_dataset

class test_shotdataset(unittest.TestCase):
    """Try instantiating a Dataset"""

    # Instantiate a resampler
    my_resampler = resampler_last(0.0, 2.0, 1e-3)

    # Instantiate a file backend
    my_backend_file = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021/")

    ds = shot_dataset(184800, MachineD3D(), [fs07, q95, neped], resampler=my_resampler, backend_file=my_backend_file, download=False, dtype=torch.float32)

    # signal_list = [fs07, ip, q95, neped]
    # my_shot = Shot(184800, MachineD3D(), signal_list)

    # # List the signals that are in the shot
    # for s in my_shot.signals:
    #     print("Signal ", s, " is of type ", type(s))


if __name__ == "__main__":
    unittest.main()

