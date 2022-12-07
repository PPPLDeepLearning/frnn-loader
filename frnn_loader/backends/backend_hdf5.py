# -*- coding: utf-8 -*-

"""Defines access to downloaded signal data using hdf5 files

This backends handles data storing and loading from HDF5 files.

The ``LocalPath`` of the `signal.yaml` files is mapped to a dataset.
"""

import logging
import numpy as np
import torch
from os import makedirs
from os.path import join, isdir
import h5py

from frnn_loader.backends.backend import backend
from frnn_loader.utils.errors import NotDownloadedError


class backend_hdf5(backend):
    """Backend that stores/loads from HDF5 files.

    Args
        root (string): Root path of the data directory
        dtype (torch.dtype, optional) : Datatype to use. Defaults to float32
    """

    def __init__(self, root, dtype=torch.float32):
        super().__init__(root, dtype)
        # Later, self.dtype is passed to hdf5 library calls. And the HDF5 library
        # only accpets numpy datatypes.
        if dtype is torch.float32:
            self.dtype = "f"

    def _mapping_path(self, sig_info, shotnr):
        return join(self.root, sig_info["Machine"])

    def load(self, sig_info, shotnr):
        """Load data.

        Args
            sig_info (string): Signal info
            shotnr (int): shot number

        Raises

            NotDownloadedError - When the HDF5 file with the shot number is not found.
        """
        map_to = self._mapping_path(sig_info, shotnr)

        # Try loading the signal from file. If we can't find the file assume we haven't
        # downloaded the data and raise the appropriate error
        fname = join(map_to, f"{shotnr}.h5")
        try:
            fp = h5py.File(fname, "r")
        except (FileNotFoundError, OSError) as e:
            err_str = f"Unable to open file {fname}"
            logging.error(err_str)
            raise NotDownloadedError(err_str)

        try:
            tb = torch.tensor(fp[sig_info["LocalPath"]]["tb"][:])
            data = torch.tensor(fp[sig_info["LocalPath"]]["zdata"][:])
        except (ValueError, KeyError) as e:
            fp.close()
            err_str = f"Unable to load timebase/signal for shot {shotnr} signal {sig_info['LocalPath']}"
            logging.error(err_str)
            raise NotDownloadedError(err_str)
        logging.info(f"Loaded signal {sig_info['Machine']}: {sig_info['Description']} from {fname}.  tb.shape = {tb.shape}, signal.shape = {data.shape}")

        return tb, data

    def store(self, sig_info, shotnr, tb, signal_data):
        """Store a signal.

        Calling store will append a dataset in the HDF5 file.

        """
        # Get mapping path
        map_to = self._mapping_path(sig_info, shotnr)

        # If the base directory does not exist, create it.
        if not isdir(map_to):
            makedirs(map_to)

        with h5py.File(join(map_to, f"{shotnr}.h5"), "a") as df:
            try:
                grp = df.create_group(sig_info["LocalPath"])
            except ValueError as err:
                logging.error(f"ValueError: {sig_info['LocalPath']} err={err}")
                raise err
            # Just assume that the signal is from MDS to construct the orgin attribute.
            try:
                grp.attrs.create(
                    "origin", f"MDS {sig_info['MDSTree']}::{sig_info['MDSPath']}"
                )
            except KeyError as err:
                # If this fails, construct the origin from the PTdata name
                grp.attrs.create("origin", f"PTDATA {sig_info['PTData']}")

            dset = grp.create_dataset("tb", tb.shape, dtype=self.dtype)
            dset[:] = tb[:]

            dset = grp.create_dataset("zdata", signal_data.shape, dtype=self.dtype)
            dset[:] = signal_data[:]
            logging.info(f"Wrote {sig_info['Description']} to {grp}.")


# end of file backend_hdf5.py
