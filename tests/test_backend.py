#-*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_backends.py
import unittest

import torch
from frnn_loader.primitives.shots import Shot
from frnn_loader.backends.machine import MachineD3D
from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.data.user_signals import q95, fs07, d3d_signals
from frnn_loader.utils.errors import NotDownloadedError


class test_backend_txt(unittest.TestCase):
    """Test routines for machines."""
    def test_backend_txt(self):
        """Test whether we can load data from the txt backend"""
        my_shot = Shot(184800, MachineD3D, [q95, fs07], torch.float32)
        my_backend = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021")

        for descr, sig in d3d_signals.items():
            try:
                tb, signal = my_backend.load(MachineD3D(), sig, my_shot.number)
            except NotDownloadedError as err:
                print(f"{err}")
                continue

            print(f"Got signal {sig}. tb.shape = ", tb.shape, ", signal.shape = ", signal.shape)

            # Let's see if any data is inf or nan
            assert(torch.any(signal == torch.inf).item() is False)
            assert(torch.any(signal == torch.nan).item() is False)


if __name__ == "__main__":
    unittest.main()

# end of file test_backends.py
