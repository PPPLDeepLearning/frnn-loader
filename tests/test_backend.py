# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_backends.py
import unittest

import torch
from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.utils.errors import NotDownloadedError


class test_backend_txt(unittest.TestCase):
    """Test routines for machines."""
    def test_backend_txt(self):
        """Test whether we can load data from the txt backend"""
        my_backend = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021")
        shotnr = 184800

        # Iterate over data that has been downloaded.
        for sig_name in ["fs07"]:
            # Instantiate a signal
            signal = signal_0d(sig_name)
            # Try using the backend to access the data for the given signal and shot
            try:
                tb, data = my_backend.load(signal.info, shotnr)
            except NotDownloadedError as err:
                print(f"{err}")
                continue

            print(f"Got signal {signal}. tb.shape = ", tb.shape, ", data.shape = ", data.shape)

            # Let's see if any data is inf or nan
            assert(torch.any(torch.isinf(data)).item() is False)
            assert(torch.any(torch.isnan(data)).item() is False)





if __name__ == "__main__":
    unittest.main()

# end of file test_backends.py
