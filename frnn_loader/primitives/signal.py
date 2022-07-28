# -*- coding: utf-8 -*-

"""
Class representations of measurement signals.

"""

from os import remove
from os.path import isfile, join, getsize

import logging

import numpy as np

from frnn_loader.utils.processing import get_individual_shot_file
from frnn_loader.utils.downloading import get_missing_value_array
from frnn_loader.utils.hashing import myhash
from frnn_loader.utils.errors import NotDownloadedError, SignalCorruptedError


class Signal:
    """Represents a signal."""

    def __init__(
        self,
        description,
        paths,
        machines,
        tex_label=None,
        causal_shifts=None,
        is_ip=False,
        normalize=True,
        data_avail_tolerances=None,
        is_strictly_positive=False,
        mapping_paths=None,
    ):
        """Initialize a signal

        Args:
            description: (string) String description of the signal
            paths: (string)
            machines: (list(machine)) List of machines on which this signal is defined
            tex_label: (string) Label used in plots
            causal_shifts: ???
            is_ip: ???
            normalize: (bool) If true, normalize this data in preprocessing. If False, skip normalization
            data_avail_tolerances:
            is_strictly_positive: (bool): If true, this data can not have negative values
            mapping_paths: ???

        Attributes:
            description: (string) String description of the signal
            paths: (string)
            machines: (list(machine)) List of machines on which this signal is defined
            tex_label: (string) Label used in plots
            causal_shifts: ???
            is_ip: ???
            normalize: (bool) If true, normalize this data in preprocessing. If False, skip normalization
            data_avail_tolerances:
            is_strictly_positive: (bool): If true, this data can not have negative values
            mapping_paths: ???


        Besides representing a signal, this class also includes methods to fetch the data.
        The methods _load_data_from_txt_safe and load_data allow to read in data from txt files.
        The method _fetch_data_basic allows to fetch the data using data access methods specified
        for the individual machines.

        """
        assert len(paths) == len(machines)
        self.description = description
        self.paths = paths
        self.machines = machines  # on which machines is the signal defined
        if causal_shifts is None:
            self.causal_shifts = [0 for m in machines]
        else:
            self.causal_shifts = causal_shifts  # causal shift in ms
        self.is_ip = is_ip
        self.num_channels = 1
        self.normalize = normalize
        if data_avail_tolerances is None:
            data_avail_tolerances = [0 for m in machines]
        self.data_avail_tolerances = data_avail_tolerances
        self.is_strictly_positive = is_strictly_positive
        self.mapping_paths = mapping_paths

    def get_file_path(self, prepath, machine, shot_number):
        """Loads signal for given machine and shot number.

        Args:
            prepath: string, Base path, conf['paths']['base_path']
            machine machine, Type of machine (D3D, NSTX, Jet...)
            shot_number: int, Unique shot identifier

        Returns:
            ???: No idea

        Constructs the filename for a signal. Format:
        prepath/machine.name/signal.dirname/shot_number
        """

        dirname = self.get_path(machine)
        return get_individual_shot_file(
            join(prepath, machine.name, dirname), shot_number
        )

    def is_valid(self, prepath, shot, dtype="float32"):
        t, data, exists = self.load_data(prepath, shot, dtype)
        return exists

    def _load_data_from_txt_safe(self, prepath, shot, dtype="float32"):
        """Safely load signal data from a stored txt file.

        Args:
            prepath:
            shot:
            dtype:

        Returns:
            data: ndarray(float) Signal datasignal

        Raises
            NotDownloadedError when the signal was not downloaded.
            SignalCorruptedError when the file has zero size, or missing_value_array was written to the file.

        This method acts as a safe wrapper around np.loadtxt and adds additional error handling.
        Bbase classes and all derived classes call this method to perform additional data
        manipulation for tasks downstream.

        """
        # Constructs the filename for a signal. Format:
        # prepath/machine.name/signal.dirname/shot_number
        file_path = get_individual_shot_file(
            join(prepath, shot.machine.name, self.get_path(shot.machine)), shot.number
        )

        # file_path = self.get_file_path(prepath, shot.machine, shot.number)
        # Make sure the file exists and has non-zero size so that we can raise
        # more specific error messages.
        if not isfile(file_path):
            raise NotDownloadedError(
                f"Signal {self.description}, shot {shot.number} was never downloaded: {file_path} does not exist"
            )

        if getsize(file_path) == 0:
            raise SignalCorruptedError(
                f"Signal {self.description}, shot {shot.number} was downloaded incorrectly (empty file). Removing."
            )

        # Load the data from a numpy file. Do not catch errors, but let them propagate through.
        data = np.loadtxt(file_path, dtype=dtype)
        if np.all(data == get_missing_value_array()):
            raise SignalCorruptedError(
                f"Signal {self.description}, shot {shot.number} contains no data"
            )

        return data

    def load_data(self, prepath, shot, dtype=np.float32):
        """Loads data from txt file and peforms data wrangling.

        Args
          prepath (str): Base path for loading
          shot (:obj:`frnn_loader.primitives.shots.Shot`): Shot
          dtype (str, optional): Data type to be used as floats.

        Returns:
            tb (ndarray) : Signal timebase
            data: ndarray(float) Signal data

        Raises
          SignalCorruptedError: When the interval where the current threshold is satisfied is too short.
                                When the time interval is too short
                                If the dynamic range of the signal is too low
                                If the timebase or the signal contains NaNs


        """
        data = self._load_data_from_txt_safe(prepath, shot, dtype)

        if np.ndim(data) == 1:
            data = np.expand_dims(data, axis=0)

        tb = data[:, 0]
        sig = data[:, 1:]

        # If desired, restrict the signal to the interval where the current threshold is satisfied.
        if self.is_ip:
            region = np.where(np.abs(sig) >= shot.machine.current_threshold)[0]
            # Raise an error if the interval is too short.
            if len(region) == 0:
                err_msg = f"Shot {shot.number}: Interval where current threshold is satisfied is too short."
                logging.error(err_msg)
                raise SignalCorruptedError(err_msg)

            first_idx, last_idx = region[0], region[-1]

            # add 50 ms to cover possible disruption event
            last_time = tb[last_idx] + 5e-2
            last_indices = np.where(tb > last_time)[0]
            if len(last_indices) == 0:
                last_idx = -1
            else:
                last_idx = last_indices[0]
            tb = tb[first_idx:last_idx]
            sig = sig[first_idx:last_idx, :]

        # make sure shot is not garbage data
        # The length should be larger than one
        # If the dynamic range of the signal is too low we assuem it is garbage data
        if tb.size <= 1:
            raise SignalCorruptedError(
                f"Signal {self.description}, shot {shot.number}: Timebase size is {tb.size}."
            )

        if tb.ndim != 1:
            raise SignalCorruptedError(
                f"Signal {self.description}: Timebase dimension is {tb.ndim}, but expected ndim=1."
            )

        # Assert that the data looks reasonable. If this is the case continue working with it
        if sig.ndim != 2:
            raise SignalCorruptedError(
                f"Signal {self.description}: Expected to have 2 dimensions but found {sig.ndim}. ??? Not sure if this has to be the case..."
            )

        if np.max(sig) - np.min(sig) < 1e-8:
            raise SignalCorruptedError(
                f"Dynamic range of signal {self.description}, shot {shot.number}, is smaller than 1e-8"
            )

        # make sure data doesn't contain nan
        if np.any(np.isnan(tb)) or np.any(np.isnan(sig)):
            raise SignalCorruptedError(
                f"Signal {self.description}, shot {shot.number} contains NaNs"
            )

        # Finall,y return time base and the signal
        return tb, sig

    def _fetch_data_basic(self, machine, shot_num, c, path=None):
        """Fetches the signal data using the machine connection.

        Args:
          machine: (machine) Machine for which to download the data
          shot_num: Shot number
          c: ???
          path: (string) Optional path to over-ride the generated signal path

        Returns:
          time: Time-base for the signal
          data: Signal data
          mapping: ???


        This method fetches the signal data using the connection mechanism for a machine.
        The function should be identical to _load_data_basic.
        The base class and all derived classes call this method to perform additional data
        manipulation for tasks downstream.
        """
        if path is None:
            path = self.get_path(machine)

        mapping = None
        time, data, mapping, success = machine.fetch_data(path, shot_num, c)
        time = np.array(time) + 1e-3 * self.get_causal_shift(machine)

        return time, np.array(data), mapping, success

    def fetch_data(self, machine, shot_num, c):
        raise DeprecationWarning("Use Signal.fetch_data!")

    def is_defined_on_machine(self, machine):
        raise DeprecationWarning("Use function body directly")
        return machine in self.machines

    def is_defined_on_machines(self, machines):
        raise DeprecationWarning("Use function body directly!")
        return all([m in self.machines for m in machines])

    def get_path(self, machine):
        idx = self.get_idx(machine)
        return self.paths[idx]

    def get_mapping_path(self, machine):
        if self.mapping_paths is None:
            return None
        else:
            idx = self.get_idx(machine)
            return self.mapping_paths[idx]

    def get_causal_shift(self, machine):
        idx = self.get_idx(machine)
        return self.causal_shifts[idx]

    def get_data_avail_tolerance(self, machine):
        idx = self.get_idx(machine)
        return self.data_avail_tolerances[idx]

    def get_idx(self, machine):
        assert machine in self.machines
        idx = self.machines.index(machine)
        return idx

    def description_plus_paths(self):
        return self.description + " " + " ".join(self.paths)

    def __eq__(self, other):
        if other is None:
            return False
        return self.description_plus_paths().__eq__(other.description_plus_paths())

    def __ne__(self, other):
        return self.description_plus_paths().__ne__(other.description_plus_paths())

    def __lt__(self, other):
        return self.description_plus_paths().__lt__(other.description_plus_paths())

    def __hash__(self):
        return myhash(self.description_plus_paths())

    def __str__(self):
        return self.description

    def __repr__(self):
        return self.description
