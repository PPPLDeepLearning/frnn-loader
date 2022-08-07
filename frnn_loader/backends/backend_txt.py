# -*- coding: utf-8 -*-

from os.path import join, getsize, isfile
import logging
import torch

from frnn_loader.utils.errors import NotDownloadedError, SignalCorruptedError


class backend_txt:
    """Backend to xtore/load from txt files.

    This backend loads downloaded data from txt files.

    Args:
        root (string) : Root path of the data directory
        dtype (torch.dtype, optional) : Datatype of the return tensor
    """

    def __init__(self, root, dtype=torch.float32):
        self.root = root
        self.dtype = dtype

    def load(self, sig_info, shotnr):
        """Loads a specified signal for a given shot on a machine.

        Args:
            sig_info (Dict) : Dictionary generated from signal yaml file
            shotnr (int) : Shot number

        Returns:
            timebase (torch.tensor) : Timebase of the signal
            signal (torch.tensor) : Signal samples

        Raises:
            ValueError: When selected signal is unavailable on the machine
            NotDownloadedError : The signal has not been downloaded
            SignalCorruptedError : The file size is zero
        """
        # Construct the path where a signal is stored for the specified machine
        # root/machine.name/signal.path/shot_number.txt
        # For this we need to pick the correct path from the signal.
        file_path = join(self.root, sig_info["Machine"], sig_info["LocalPath"], f"{shotnr}.txt")

        # Perform checks to see that the file is good.
        if not isfile(file_path):
            raise NotDownloadedError(
                f"Signal {sig_info['Description']}, shot {shotnr} was never downloaded: {file_path} does not exist"
            )

        if getsize(file_path) == 0:
            raise SignalCorruptedError(
                f"Signal {sig_info['description']}, shot {shotnr} has size==0. Removing."
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


    def store(self, sig_info, shot, data):
        pass


# end of file backend_txt.py
