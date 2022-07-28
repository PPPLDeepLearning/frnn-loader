# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import logging

from frnn_loader.utils.errors import SignalCorruptedError


class shot_dataset(Dataset):
    """Dataset representing a shot.

    Args:
        shotnr (int) :
        machine (:obj:`frnn_loader.backend.machines.machine`)
        signal_list (list) :
        dtype (torch.dtype, optional) : Floating-point
        backend ()
        download (bool) : If True, use

    Attributes:
        signals_dict (dict) : Dictionary with signals as keys and pre-processed data array of the signal as values
    """

    def __init__(self, shotnr, machine, signal_list, dtype, backend, download=False):
        """Initializes the shot dataset."""

        self.shotnr = shotnr
        self.machine = machine
        self.signal_list = signal_list
        self.dtype = dtype
        self.backend = backend
        self.download = download

        self.signals_dict = {}
        pass

    def _preprocess(self):
        """Loads all signals and resamples to common time-base.

        This function provides the main interface to make shot data available
        inside the object.
        First, each signal defined in the signal is loaded.
        With all signals at hand, a common time-base can be defined.
        Then the signals are clipped to this timebase and resampled
        with the sampling frequency specified in the config object.

        Parameters:
            conf (dict): global configuration

        Returns:
            None
        """

        pass

    def _load_signal_data(self, conf):
        """Load signals and time bases for a given shot.

        This method loads all signal data and time bases from text files.
        Used keys from conf:
        conf["paths"]["data"] - ???
        conf["data"]["floatx"] - Datatype, either float32 or float64
        conf["paths"]["signal_prepath"] - ???

        Parameters:
          conf (dict) global configuration


        Output:
          time_arrays: List of time bases for the specified signals
          signal_arrays: List of ndarrays for the specified signals
          t_min: Min time of all time bases
          t_max: Max time in all time bases

        Raises:

            BadShotException: Multiple possibilities

        """
        t_min = -torch.inf  # Smallest time in all time bases
        t_max = torch.inf  # Largest time in all time bases
        # t_thresh = -1
        signal_arrays = []  # To be populated with a list of ndarrays with shot data
        tb_arrays = []  # To be populated with a list of ndarrays, containing time bases

        invalid_signals = 0  # Counts the number of invalid signals for this shot
        signal_prepath = conf["paths"]["signal_prepath"]

        # Iterate over all signals and extract data
        for signal in self.signals:
            # Try loading the signal. When this fails, append dummy data.
            try:
                tb, signal = signal.load_data(signal_prepath, self, self.dtype)
            except SignalCorruptedError as err:
                # TODO: Why is there a sig[1] in the dimension
                # signal = np.zeros((tb.shape[0], sig[1]))
                logging.error(f"Erorr occorued: {err}")
                invalid_signals += 1
                signal_arrays.append(np.zeros([tb.shape[0], 1]), dtype=self.dtype)
                tb_arrays.append(np.arange(0, 20, 1e-3, dtype=self.dtype))

            # At this point we assume that we have good data from the shot.

            # Update t_min and append signal and timebase to the working data
            t_min = max(t_min, tb.min())
            signal_arrays.append(signal)
            tb_arrays.append(tb)

            # Handle edge-case where the shot is supposedly disruptive, but the disruption
            # happens after the signal ends.
            # In the case where no previously added diagnostic covers the disruption we raise and error
            # If a previously added diagnostic covers the disruption we add dummy data for this signal
            if self.is_disruptive and self.t_disrupt > tb.max():
                t_max_total = tb.max() + signal.get_data_avail_tolerance(self.machine)

                if self.t_disrupt > t_max_total:
                    # The time of the disruption is out of range.
                    # Instead of the signal we use dummy data.
                    invalid_signals += 1
                    tb = np.arange(0, 20.0, 1e-3)
                    signal = np.zeros((tb.shape[0], signal.shape[1]))
                    # Setting the entire channel to zero to prevent any peeking into possible disruptions from this early ended channel
                else:
                    t_max = tb.max() + signal.get_data_avail_tolerance(self.machine)
            else:
                t_max = min(t_max, tb.max())

        # make sure the shot is long enough.
        # dt = conf["data"]["dt"]

        # Perform sanity checks.
        # 1/ t_max should be larger than t_min
        if t_max < t_min:
            raise BadShotException(
                f"Shot {self.number} has t_max = {t_max} < t_min = {t_min}. Expected t_max > t_min."
            )
        # Remove this test so we don't have to pass conf for now
        #        # 2/ The shot timebase should be sufficiently long enough.
        #        if (t_max - t_min)/dt <= (2 * conf['model']['length'] + conf['data']['T_min_warn']):
        #            raise BadShotException(f"Shot {self.number} contains insufficient data, omitting.")
        # 3/ The shot is marked disruptive and the disruption occurs after all measurements
        if self.is_disruptive and self.t_disrupt > t_max:
            raise BadShotException(
                f"Shot {self.number} is disruptive at {self.t_disrupt}s but data stops at {t_max}"
            )

        if invalid_signals > 2:
            raise BadShotException(f"Shot {self.number} has more than 2 bad channels.")

        if self.is_disruptive:
            t_max = self.t_disrupt

        return tb_arrays, signal_arrays, t_min, t_max

    def __len__(self):
        """Number of time samples in the dataset."""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """Fetches a single item."""
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# end of file frnn_dataset.py
