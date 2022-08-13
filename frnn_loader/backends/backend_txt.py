# -*- coding: utf-8 -*-

"""Defines access to downloaded signal data using txt files.

    This backend loads downloaded data from txt files. 
    Rows in the txt files correspond to time records.
    The first column corresponds to the time of the record.
    The following columns correspond to individual channels of the signal.

    File paths for individual signals are constructed as
    root/MMM/LocalPath/shotnr.txt.

    Here
    - root is the root member of the class
    - MMM is the machine specified for a signal
    - LocalPath is the LocalPath specified in the signal in `d3d_signals.yaml`
    - shotnr is the Shot number.
"""
from os import path as path
from os import makedirs
import logging
import torch

from frnn_loader.backends.backend import backend
from frnn_loader.utils.errors import NotDownloadedError, SignalCorruptedError


class backend_txt(backend):
    """Backend to store/load from txt files.


    Args:
        root (string) : Root path of the data directory
        dtype (torch.dtype, optional) : Datatype of the return tensor
    """

    def __init__(self, root, dtype=torch.float32):
        super().__init__(root, dtype)

    def load(self, sig_info, shotnr):
        """Loads a specified signal for a given shot on a machine.

        Args
            sig_info (Dict) : Dictionary generated from signal yaml file
            shotnr (int) : Shot number

        Returns
            timebase (torch.tensor) : Timebase of the signal
            signal (torch.tensor) : Signal samples

        Raises
            ValueError: When selected signal is unavailable on the machine
            NotDownloadedError : The signal has not been downloaded
            SignalCorruptedError : The file size is zero
        """
        # Construct the path where a signal is stored for the specified machine
        # root/machine.name/signal.path/shot_number.txt
        # For this we need to pick the correct path from the signal.
        file_path = path.join(
            self._construct_file_path(sig_info, shotnr), f"{shotnr}.txt"
        )

        # Perform checks to see that the file is good.
        if not path.isfile(file_path):
            raise NotDownloadedError(
                f"Signal {sig_info['Description']}, shot {shotnr} was never downloaded: {file_path} does not exist"
            )

        if path.getsize(file_path) == 0:
            raise SignalCorruptedError(
                f"Signal {sig_info['Description']}, shot {shotnr} has size==0. Removing."
            )

        # Load manually into a list and convert to torch.tensor
        float_vals = []
        with open(file_path, "r") as fp:
            for line in fp.readlines():
                float_vals.append([float(val) for val in line.split()])
        data = torch.tensor(float_vals, dtype=self.dtype)
        # print(f"... In load. data.shape = ", data.shape)
        # First column is the timebase
        # After second column is the signal data
        return data[:, 0], data[:, 1:]

    def store(self, sig_info, shotnr, tb, signal):
        """Store a signal.

        Args:
            sign_info (dict) : Dictionary generated from signal yaml file
            shotnr (int): Shot number
            tb (torch.tensor): Timebase of the signal
            signal (torch.tensor): Signal samples

        Returns:
            None
        """
        # Concatenate time-base and signal sample tensor
        all_vals = torch.cat([tb.unsqueeze(1), signal], 1)
        file_path = path.join(
            self._construct_file_path(sig_info, shotnr), f"{shotnr}.txt"
        )

        # If the directory does not exist yet create it
        if not path.isdir(path.dirname(file_path)):
            makedirs(path.dirname(file_path))
            logging.info(f"backend_txt: creating {path.dirname(file_path)}")

        with open(file_path, "w") as fp:
            for row in range(all_vals.shape[0]):
                # Write first number as floating point, the others in
                # scientific notation
                line = f"{all_vals[row, 0]:08f} "
                line += " ".join([f"{val:10.8e}" for val in all_vals[row, 1:]])
                line += "\n"
                fp.write(line)


# end of file backend_txt.py
