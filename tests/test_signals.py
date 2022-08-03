# -*- coding: utf-8 -*-
import unittest

from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.backends.machine import MachineD3D
from frnn_loader.primitives.signal import signal_0d

from frnn_loader.data.user_signals import d3d_signals_0D


class TestSignals(unittest.TestCase):
    """Test routines for machines."""

    def test_signal_fs07(self):
        """Test whether we can instantiate fs07 signal"""
        my_backend = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021")
        fs07 = signal_0d("filterscope fs07", ['fs07'], [MachineD3D])

        fs07.load_data(184800, MachineD3D(), my_backend)

    def test_signals_0d_d3d(self):
        """Test whether we can instantiate all 0d signals D3D signals."""
        for key, signal in d3d_signals_0D.items():
            print(f"Testing signal {signal}", type(signal))

    def test_signals_1d_d3d(self):
        """Test whether we can instantiate all 1D signals at D3D."""
        pass



if __name__ == "__main__":
    unittest.main()

# end of file test_signals.py