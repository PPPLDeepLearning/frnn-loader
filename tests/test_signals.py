#-*- coding: utf-8 -*-
import unittest

from frnn_loader.primitives.machine import  MachineD3D
from frnn_loader.primitives.signal import Signal


class TestSignals(unittest.TestCase):
    """Test routines for machines."""
    def test_signal_fs07(self):
        """Test whether we can instantiate signals"""
        fs07 = Signal("filterscope fs07", [fs07'], [MachineD3D])


if __name__ == "__main__":
    unittest.main()

# end of file test_signals.py