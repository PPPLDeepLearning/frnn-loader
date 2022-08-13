# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_frnndataset.py
import unittest
import logging
import tempfile
from os import makedirs

from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.utils.errors import NotDownloadedError

FORMAT = "%(asctime)s unittest test_frnndataset %(message)s"
logging.basicConfig(format=FORMAT,level=logging.DEBUG)

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
        """Compare downloaded signal shape to data read from backend."""
        shotnr = 180619
        sig_names = ["dens", "fs07", "q95", "qmin", "li", "ip", "betan",
                     "energy", "lm", "pradcore", "pradedge", "bmspinj", "bmstinj",
                     "iptdirect", "iptarget",
                     "tmamp1", "tmamp2", "tmfreq1", "tmfreq2"]

        # The working dir should be empty. Test will fail if not empty.
        wd = tempfile.mkdtemp(dir="/home/rkube/tmp/")

        my_backend = backend_txt(wd)
        my_fetcher = fetcher_d3d_v1()

        for name in sig_names:
            logging.info(f"========================= {name} =================================")
            signal = signal_0d(name)
            # Download the signal
            xdata, _, zdata, _, _, _ = my_fetcher.fetch(signal.info, shotnr)
            # Write it to file using the backend.
            my_backend.store(signal.info, shotnr, xdata, zdata)

            # Now read it again using the same backend.
            tb, sig_data = my_backend.load(signal.info, shotnr)

            assert((tb - xdata).mean() < 1e-8)
            assert((sig_data - zdata).mean() < 1e-8)


if __name__ == "__main__":
    unittest.main()

# end of file test_signals.py
