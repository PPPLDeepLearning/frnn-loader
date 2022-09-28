# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_frnndataset.py
import unittest
import logging
import tempfile
import shutil
import errno
from os import environ

from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.utils.errors import NotDownloadedError

FORMAT = "%(asctime)s unittest test_frnndataset %(message)s"
logging.basicConfig(format=FORMAT,level=logging.DEBUG)

class TestSignals(unittest.TestCase):
    """Test routines for machines."""
    @classmethod
    def setUpClass(cls):
        """Set up signals tests
        * Define a temp directory
        """
        # The working dir should be empty. Test will fail if not empty.
        try:
            cls.root = tempfile.mkdtemp(dir=environ["TMPDIR"])
        except KeyError:
            cls.root = tempfile.mkdtemp(dir="/home/rkube/tmp/")

        print(f"cls.root = {cls.root}")
        cls.shotnr = 180619

    @classmethod
    def tearDownClass(cls):
        """Tear down signal tests.
        
        * Remove temp directory."""
        try:
            shutil.rmtree(cls.root)  # delete directory
        except OSError as exc:
            if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                raise  # re-raise exception


    def test_signals_0d(self):
        """Compare downloaded signal shape to data read from backend."""
        sig_names = ["dssdenest", "fs07", "q95", "qmin", "efsli", "ipspr15V", "efsbetan",
                     "efswmhd", "dusbradial", "echpwrc", "pradcore", "pradedge", "bmspinj", "bmstinj",
                     "iptdirect", "ipsiptargt", "ipeecoil",
                     "tmamp1", "tmamp2", "tmfreq1", "tmfreq2"]

        my_backend = backend_txt(self.root)
        my_fetcher = fetcher_d3d_v1()

        for name in sig_names:
            signal = signal_0d(name)
            # Download the signal
            xdata, _, zdata, _, _, _ = my_fetcher.fetch(signal.info, self.shotnr)
            # Write it to file using the backend.
            my_backend.store(signal.info, self.shotnr, xdata, zdata)

            # Now read it again using the same backend.
            tb, sig_data = my_backend.load(signal.info, self.shotnr)

            assert((tb - xdata).mean() < 1e-8)
            assert((sig_data - zdata).mean() < 1e-8)


if __name__ == "__main__":
    unittest.main()

# end of file test_signals.py
