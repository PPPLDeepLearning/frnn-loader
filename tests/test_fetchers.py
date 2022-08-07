# -*- coding: utf-8 -*-
# run as
# python -m unittest tests/test_fetchers.py

import unittest

from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.backends.machine import MachineD3D
from frnn_loader.data.user_signals import d3d_signals_0D


class test_fetch_d3d(unittest.TestCase):
    """Try fetching from D3D - atlas.gat.com"""
    print("Testing D3D")
    fetcher = fetcher_d3d_v1()

    # Iterate over all 0d signals and see if we can fetch them from MDS
    for key, signal in d3d_signals_0D.items():
        print(f"Testing signal {key}")
        xdata, data, ydata = fetcher.fetch(signal.paths[signal.machines.index(MachineD3D)], 184800)


if __name__ == "__main__":
    unittest.main()