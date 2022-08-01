# -*- coding: utf-8 -*-

from os.path import join, getsize, isfile
import logging
import torch

from frnn_loader.utils.errors import NotDownloadedError, SignalCorruptedError


class backend_txt:
    """Backend to load from txt files.

    This backend loads downloaded data from txt files.

    Args:
        root (string) : Root path of the data directory
        dtype (torch.dtype, optional) : Datatype of the return tensor
    """

    def __init__(self, root, dtype=torch.float32):
        self.root = root
        self.dtype = dtype

    def load(self, machine, sig, shotnr):
        """Loads a specified signal for a given shot on a machine.

        Args:
            machine (machine) : Which machine to load for
            sig (signal) : Type of signal to load
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
        try:
            base_path = join(
                self.root, machine.name, sig.paths[sig.machines.index(machine)]
            )
        except ValueError as err:
            logging.error(
                "Error fetching signal {sig} for machine {machine}, shotnr {shotnr}: {err}"
            )
            raise err

        file_path = join(base_path, f"{shotnr}.txt")

        # Perform checks to see that the file is good.
        if not isfile(file_path):
            raise NotDownloadedError(
                f"Signal {sig.description}, shot {shotnr} was never downloaded: {file_path} does not exist"
            )

        if getsize(file_path) == 0:
            raise SignalCorruptedError(
                f"Signal {sig.description}, shot {shotnr} was downloaded incorrectly (empty file). Removing."
            )

        # Load manually into a list and convert to torch.tensor
        float_vals = []
        with open(file_path, "r") as fp:
            for line in fp.readlines():
                float_vals.append([float(val) for val in line.split()])
        data = torch.tensor(float_vals, dtype=self.dtype)
        #print(f"... In load. data.shape = ", data.shape)
        # First column is the timebase
        # After second column is the signal data
        return data[:, 0], data[:, 1:]


# end of file backend_txt.py
