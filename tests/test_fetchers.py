# -*- coding: utf-8 -*-
# run as
# python -m unittest tests/test_fetchers.py

"""Test routines for fetcher classes.

fetchers are used by signals to download signal data from remote servers.

The unit tests here test 
"""

import unittest
import torch

from frnn_loader.primitives.signal import signal_0d
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.utils.errors import BadDownloadError


class test_fetch_d3d(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        """Set up fetcher tests.
        
        * Define a shot number
        * Define signals to use
        """
        cls.shotnr = 180619
        cls.signal_list = ["fs07", "q95"]

    """Try fetching from D3D - atlas.gat.com"""
    def test_fetch_d3d(self):
        fetcher = fetcher_d3d_v1()
        # Iterate over data that has been downloaded.
        for sig_name in self.signal_list:
            # Instantiate a signal
            signal = signal_0d(sig_name)
            # Try using the backend to access the data for the given signal and shot
            try:
                xdata, ydata, zdata, xunits, yunits, zunits = fetcher.fetch(signal.info, self.shotnr)
            except BadDownloadError as err:
                print(f"{err}")
                continue
            
            # Let's see if any data is inf or nan
            assert(torch.any(torch.isinf(zdata)).item() is False)
            assert(torch.any(torch.isnan(zdata)).item() is False)


if __name__ == "__main__":
    unittest.main()

# end of file test_fetchers.py
