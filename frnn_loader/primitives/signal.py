# -*- coding: utf-8 -*-

"""
Class representations of measurement signals.

"""

from pathlib import Path
from os.path import join
import logging
import yaml
from frnn_loader.utils.errors import NotDownloadedError, SignalCorruptedError


class signal_base:
    """Abstract base class for all signal.

    The signal_base class defines a mapping from the defition of a single to
    the actual data. Definition of signals are stored in yaml files, see f.ex.
    frnn_loader/data/d3d_signals.yaml. As an example, this  definition looks like

    ```
        dssdenest:
          Machine: D3D
          Description: Plasma Density
          PTData: dssdenest
          LocalPath: dssdenest
          ndim: 0
          Channels: 1
    ```

    This defines the signals `dssdenest` as being defined for D3D, with source `PTDATA` and point name
    `dssdenest`. It will be stored locally under the path `dssdenest`. It is a scalar time-series,
    `ndim=0`, and occupies one channel.

    The class `signal_base` only ties together definitions of signals and how they are fetched and stored.
    Fetching and loading themself are implemented through fetchers and backends.


    Args:

        tag (str): Key name for the signal. This will be matched with to a key defined in user_signal.yaml
        sig_def_fname (str): Path to definitions of user signals
        root (str): Root directory where to search for signal definition yaml files

    """

    def __init__(self, tag, sig_def_fname="d3d_signals.yaml", root=None):
        # If no root path is given, we just use the path to the data files in the repo
        if root is None:
            base_path = Path(__file__).parent
            self.root = join(base_path, "..", "data")
        else:
            self.root = root

        sig_def_fname = join(self.root, sig_def_fname)

        self.tag = tag
        with open(sig_def_fname, "r") as df:
            signal_defs = yaml.load(df, Loader=yaml.FullLoader)
        self.info = signal_defs[tag]

    def load_data(self, shotnr, backend):
        """Load data using the backend.

        Args:
            shotnr (int) : Shot number
            backend (file_backend) : Backend to use for loading

        Returns:
            timebase (torch.tensor) : Time base for signal. Dim0: sample (time)
            signal (torch.tensor) : Sampled signal Dim0: time, dim1: channel

        Raises:
            Propagate exceptions from the load method of the backend
        """
        # Use the backend to fetch the data from file or whatever
        try:
            tb, data = backend.load(self.info, shotnr)
        except Exception as err:
            logging.error(f"{err}")
            raise err

        return tb, data

    def fetch_data(self, shotnr, fetcher):
        """Fetch data using a data_fetcher.

        Args:
            shotnr (int): Shot number
            fetcher (fetcher): Datafetcher to use

        Returns:
            timebase (torch.tensor): Time base for the signal. dim0: sample (time)
            signal (torch.tensor): Sampels signal.
        """
        return fetcher.fetch(self.info, shotnr)

    def __eq__(self, other):
        assert isinstance(other, signal_base)
        return self.__str__().__eq__(other.__str__())

    def __ne__(self, other):
        assert isinstance(other, signal_base)
        return self.__str__().__ne__(other.__str__())

    def __str__(self):
        return f"{self.info['Machine']}: {self.info['Description']}"

    def __repr__(self):
        return self.__str__()


class signal_0d(signal_base):
    """Scalar signal

    Args:
        tag (str): Key name for dictionary defined in user_signal.yaml
        sig_def_fname (str): Signal definition file to use.

    """

    def __init__(self, tag, sig_def_fname="d3d_signals.yaml"):
        super().__init__(tag, sig_def_fname)
        assert self.info["ndim"] == 0
        self.num_channels = 1
        # I'm skipping several parameters from the original definition here.


class signal_1d(signal_base):
    """Scalar signal

    Args:
        tag (str): Key name for dictionary defined in user_signal.yaml
        sig_def_fname (str): Signal definition file to use.

    """

    def __init__(self, tag, sig_def_fname="d3d_signals.yaml"):
        super().__init__(tag, sig_def_fname)
        assert self.info["ndim"] == 1
        self.num_channels = self.info["Channels"]
        # I'm skipping several parameters from the original definition here.
