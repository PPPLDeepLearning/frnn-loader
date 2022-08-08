# -*- coding: utf-8 -*-
import unittest

from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.utils.errors import NotDownloadedError

#from frnn_loader.data.user_signals import d3d_signals_0D


class TestSignals(unittest.TestCase):
    """Test routines for machines."""

    def test_signal_fs07(self):
        """Test whether we can instantiate fs07 signal"""
        my_backend = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021")
        #fs07 = signal_0d("filterscope fs07", ['fs07'], [MachineD3D])
        # Instantiate a 0d signal.
        fs07 = signal_0d("fs07")
        fs07.load_data(184800, my_backend)

    def test_signals_0d(self):
        # Bad: pradedge
        sig_names = ["dens", "fs07", "q95", "qmin", "li", "ip", "betan",
                     "energy", "lm", "pradcore", "pradedge", "bmspinj", "bmstinj",
                     "iptdirect", "iptarget",
                     "tmamp1", "tmamp2", "tmfreq1", "tmfreq2",
                     "etemp_profile"]
        my_backend = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021")
        my_fetcher = fetcher_d3d_v1()

        for name in sig_names:
            print(f"========================= {name} =================================")
            signal = signal_0d(name)
            bad_size = True
            try:
                signal.load_data(186019, my_backend)
            except NotDownloadedError as err:
                dl = signal.fetch_data(186019, my_fetcher)
                print(f"Downloaded time series with {dl[2].size} elements")
                print(f"Max: {dl[2].max()}, min: {dl[2].min()}, mean: {dl[2].mean()}, std: {dl[2].std()}")




    # def test_signals_0d_d3d(self):
    #     """Test whether we can instantiate all 0d signals D3D signals."""
    #     for key, signal in d3d_signals_0D.items():
    #         print(f"Testing signal {signal}", type(signal))

    # def test_signals_1d_d3d(self):
    #     """Test whether we can instantiate all 1D signals at D3D."""
    #     pass


if __name__ == "__main__":
    unittest.main()

# end of file test_signals.py
