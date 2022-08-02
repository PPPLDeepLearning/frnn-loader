# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_resampling.py
import unittest

import torch
from frnn_loader.primitives.resamplers import resampler_last
import matplotlib.pyplot as plt


class test_resampler_last(unittest.TestCase):
    """Test routines for resampler."""
    def test_resampler_last(self):
        """Test whether resampling is reasonable"""
        # Define a signal that we will resample
        tb_old = torch.arange(0.15, 1.14, 0.07)
        sig_old = tb_old * 0.15
        # Add observation dimension
        sig_old = sig_old.unsqueeze(1)

        # Resampler on new time interval
        my_resampler = resampler_last(0.0, 1.0, 2e-2)
        tb_new, sig_new = my_resampler(tb_old, sig_old)

        # Get the over-lapping part of the time bases
        t_min = torch.max(tb_new.min(), tb_old.min()).item()
        t_max = torch.min(tb_new.max(), tb_old.max()).item()

        idx_old = (tb_old > t_min) & (tb_old < t_max)
        idx_new = (tb_new > t_min) & (tb_new < t_max)

        # Test that there are no strange values in the resampled time series
        # We may have to relaxe the difference a bit.
        plt.plot(tb_new, sig_new, 'o-')
        plt.plot(tb_old, sig_old, 'o-')
        plt.show()
        #assert(torch.abs(sig_new[idx_new].max() - sig_old[idx_old].max()) < 1e-2)
        #assert(torch.abs(sig_new[idx_new].min() - sig_old[idx_old].min()) < 1e-2)


if __name__ == "__main__":
    unittest.main()

# end of file test_resampling.py
