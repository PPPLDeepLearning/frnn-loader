# -*- coding: utf-8 -*-
# run with
# python -m unittest tests/test_backends.py
import unittest
import tempfile
import shutil


import torch
from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.backends.backend_hdf5 import backend_hdf5
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.utils.errors import NotDownloadedError, BadDownloadError

unittest.TestLoader.sortTestMethodsUsing = None

class test_backends(unittest.TestCase):
    """Test routines for machines."""
     
    @classmethod
    def setUpClass(cls):
        """Set up unit backend tests.
        
        * Define a temp directory
        * Define a shot
        * Define a list of signals to use
        """
        cls.root = tempfile.mkdtemp(dir="/home/rkube/tmp/")
        cls.shotnr = 180619
        cls.signal_list = ["fs07", "q95"]

        print(f"-------- setup(): self.root = {cls.root}")

    
    @classmethod
    def tearDownClass(cls):
        """Tear down unti backend tests.
        
        * Delete temp directory.
        """
        pass
        # try:
        #     shutil.rmtree(cls.root)  # delete directory
        # except OSError as exc:
        #     if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
        #         raise  # re-raise exception

    

    def test_backend_txt(self):
        """Store and successively load data from the txt backend."""
        fetcher = fetcher_d3d_v1()
        my_backend = backend_txt(self.root)

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

            my_backend.store(signal.info, self.shotnr, xdata, zdata)

        # Iterate over data that has been downloaded.
        for sig_name in self.signal_list:
            # Instantiate a signal
            signal = signal_0d(sig_name)
            # Try using the backend to access the data for the given signal and shot
            try:
                tb, data = my_backend.load(signal.info, self.shotnr)
            except NotDownloadedError as err:
                print(f"{err}")
                continue

            # Let's see if any data is inf or nan
            assert(torch.any(torch.isinf(tb)).item() is False)
            assert(torch.any(torch.isnan(tb)).item() is False)

            # Let's see if any data is inf or nan
            assert(torch.any(torch.isinf(data)).item() is False)
            assert(torch.any(torch.isnan(data)).item() is False)


    def test_backend_store_hdf5(self):
        """Store and successively load data from HDF5 backend."""
        fetcher = fetcher_d3d_v1()
        my_backend = backend_hdf5(self.root)

        for sig_name in self.signal_list:
            # Instantiate a signal
            signal = signal_0d(sig_name)
            # Try using the backend to access the data for the given signal and shot
            try:
                xdata, ydata, zdata, xunits, yunits, zunits = fetcher.fetch(signal.info, self.shotnr)
            except BadDownloadError as err:
                print(f"{err}")
                continue

            my_backend.store(signal.info, self.shotnr, xdata, zdata)

        # Iterate over signals and load
        for sig_name in self.signal_list:
            # Instantiate a signal
            signal = signal_0d(sig_name)
            try:
                tb, data = my_backend.load(signal.info, self.shotnr)

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
