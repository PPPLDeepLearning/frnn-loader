# -*- coding: utf-8 -*-
# run as 
# python -m unittest tests/test_transform.py


"""Test predicting disruption time

Shots are provided in tabular form. The first column gives a shot number
The second column specifies the disruption time. 
If the shot is non-disruptive, the disruption time is -1.0

Examples

167494   3.478500
167575   -1.000000

Lists of these files are stored on the PU systems under:
/projects/FRNN/shot_lists

Here we take the D3d benchmark:
d3d_clear_100.txt
d3d_disrupt_100.txt

"""

from os import environ
from os.path import join
import unittest
import tempfile
import logging
import torch

from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.resamplers import resampler_causal
from frnn_loader.backends.backend_hdf5 import backend_hdf5
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.loaders.frnn_dataset_disk import shot_dataset_disk
from frnn_loader.utils.errors import BadDownloadError

FORMAT = "%(asctime)s unittest test_transform %(message)s"
logging.basicConfig(format=FORMAT,level=logging.DEBUG)


class test_transform_d3d_test100(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up fetcher tests

        - Defines the shot lists
        - Defines signals to use
        """

        try:
            cls.root = tempfile.mkdtemp(dir=environ["TMPDIR"])
        except KeyError:
            cls.root = tempfile.mkdtemp(dir="/home/rkube/tmp/")

        cls.basedir = "/projects/FRNN/shot_lists"
        cls.shotlist_clear = []
        with open(join(cls.basedir, "d3d_clear_100.txt"), "r") as fp:
            for line in fp.readlines():
                # Convert shotnr to int and ttd to float
                shotnr, ttd = [trf(val) for trf, val in zip([int, float], line.split())]
                cls.shotlist_clear.append(shotnr)

                break

        cls.shotlist_disrupt = []
        cls.ttd_list = []
        with open(join(cls.basedir, "d3d_disrupt_100.txt"), "r") as fp:
            for line in fp.readlines():
                # Convert shotnr to int and ttd to float
                shotnr, ttd = [trf(val) for trf, val in zip([int, float], line.split())]
                cls.shotlist_disrupt.append(shotnr)
                cls.ttd_list.append(ttd)

                break

        print("shotlist_clear = ", cls.shotlist_clear)
        print("shotlist_disrupt = ", cls.shotlist_disrupt)
        print("ttd = ", cls.ttd_list)

    def test_transform(self):
        """Instantiate datasets that include TTD transformation"""

        my_resampler = resampler_causal(0.0, 2e3, 1e0)

        # Instantiate a file backend
        my_backend_file = backend_hdf5("/home/rkube/datasets/frnn/")
        my_fetcher = fetcher_d3d_v1()
        root = self.root

        signal_fs07 = signal_0d("fs07")
        signal_q95 = signal_0d("q95")
        signal_pinj = signal_0d("bmspinj")

        ds_clear_list = []
        for shotnr in self.shotlist_clear:
            ds = shot_dataset_disk(shotnr, 
                                   predictors=[signal_fs07, signal_q95], 
                                   resampler=my_resampler, 
                                   backend_file=my_backend_file, 
                                   fetcher=my_fetcher, 
                                   root=root,
                                   download=True,
                                   dtype=torch.float32)

            ds_clear_list.append(ds)

        


if __name__ == "__main__":
    unittest.main()