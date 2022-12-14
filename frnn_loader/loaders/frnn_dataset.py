# -*- coding: utf-8 -*-

"""FRNN Dataset

This is a doc string for the class.

"""


import torch
from torch.utils.data import Dataset
import logging

from frnn_loader.primitives.signal import signal_0d
from frnn_loader.utils.errors import (
    NotDownloadedError,
    SignalCorruptedError,
    BadShotException,
)


class shot_dataset(Dataset):
    """Dataset representing a shot.

    A dataset provides access to multiple signals for a shot
    through the `__getitem__` method.

    Args
        shotnr (int) :
        predictors (list) : List of signals
        resampler (resampler) : Re-sampler to use
        backend_file (backend) : Backend to use for loading data from the file system
        backend_fetcher (fetcher) : Data fetcher to use when downloading data
        download (bool) : If True, download missing data using a data fetcher
        dtype (torch.dtype, optional) : Floating-point

    Attributes
        signals_dict (dict) : Dictionary with signals as keys and pre-processed data array of the signal as values
    """

    def __init__(
        self,
        shotnr,
        predictors,
        resampler,
        backend_file,
        backend_fetcher=None,
        download=False,
        dtype=torch.float32,
    ):
        """Initializes the shot dataset."""
        self.shotnr = shotnr
        self.predictors = predictors
        self.resampler = resampler
        self.backend_file = backend_file
        self.backend_fetcher = backend_fetcher
        self.download = download
        self.dtype = dtype

        # If want to download, see that we have a fetcher passed
        # TODO: Make this nicer.
        if self.download:
            assert self.backend_fetcher is not None

        # Construct a tensor that will hold the data.
        # The dimensions consist of sum of channels over all predictors and
        # the length of the time-interval that we resample to.
        self.total_channels = sum([sig.num_channels for sig in self.predictors])
        self.signal_tensor = torch.zeros(
            (len(self.resampler), self.total_channels), dtype=self.dtype
        )

        self._preprocess()

    def _preprocess(self):
        """Loads all signals and resamples to common time-base.

        This function provides the main interface to make shot data available
        inside the object.
        First, each signal defined in `self.predictors` is loaded.

        With all signals at hand, a common time-base can be defined.
        Then the signals are clipped to this timebase and resampled
        with the sampling frequency specified in the config object.

        Parameters
            conf (dict): global configuration

        Returns
            None
        """
        logging.info(f"Preprocessing shot {self.shotnr}")
        time_arrays, signal_arrays, t_min, t_max = self._load_signal_data()

        assert len(signal_arrays) > 0
        assert len(signal_arrays) == len(time_arrays)
        assert len(signal_arrays) == len(self.predictors)

        # Re-sample each signal
        # Store the re-sampled signal in a tensor
        # Keep track of how many channels a signal has used
        curr_channel = 0
        for (i, signal) in enumerate(self.predictors):
            logging.debug(f"Resampling signal {signal}")
            # Cut the signal to [t_min:t_max]
            good_idx = (time_arrays[i] >= t_min) & (time_arrays[i] <= t_max)
            tb = time_arrays[i][good_idx]
            # We can generally address signals as [sample_idx, channel_idx], since even
            # 0d signals are 2d tensors.
            sig = signal_arrays[i][good_idx, :]
            # Interpolate on new time-base
            tb_rs, sig_rs = self.resampler(tb, sig)

            # Populate signals_tensor with the re-sampled signals
            self.signal_tensor[
                :, curr_channel : curr_channel + signal.num_channels
            ] = sig_rs[:]
            curr_channel += signal.num_channels
        # Store the time-base

        self.tb = tb_rs

        return None

    def _load_signal_data(self):
        """Load signals and time bases for a given shot.

                This happens either through the provided file backend.
        If this fails and download is True, the provided fetcher will be
        used to download the file.

        Args

        Output
          time_arrays: List of time bases for the specified signals
          signal_arrays: List of ndarrays for the specified signals
          t_min: Min time of all time bases
          t_max: Max time in all time bases

        Raises
            BadShotException:
                * If t_max > t_min.
                * If there are multiple bad channels across the diagnostics.

        """
        t_min = -1e9  # Smallest time in all time bases
        t_max = 1e9  # Largest time in all time bases
        # t_thresh = -1
        signal_arrays = []  # To be populated with a list of ndarrays with shot data
        tb_arrays = []  # To be populated with a list of ndarrays, containing time bases
        invalid_signals = 0  # Counts the number of invalid signals for this shot

        # Iterate over all signals and extract data
        for signal in self.predictors:
            # Try loading the signal. When this fails, append dummy data.
            try:
                tb, signal_data = self.backend_file.load(signal.info, self.shotnr)

            except SignalCorruptedError as err:
                # TODO: Why is there a sig[1] in the dimension
                # signal = np.zeros((tb.shape[0], sig[1]))
                logging.error(f"SignalCorrupted occured: {err}")
                invalid_signals += 1
                signal_arrays.append(torch.zeros([tb.shape[0], 1]), dtype=self.dtype)
                tb_arrays.append(torch.arange(0.0, 1e3, 1e0, dtype=self.dtype))

            except NotDownloadedError as err:
                logging.error(f"Signal not downloaded: {err}")
                if self.download:
                    logging.info(f"Downloading signal {signal}")
                    # TODO: Download signal
                    tb, _, signal_data, _, _, _ = self.backend_fetcher.fetch(
                        signal.info, self.shotnr
                    )
                    self.backend_file.store(signal.info, self.shotnr, tb, signal_data)

                else:
                    raise err

            logging.info(
                f"Loaded signal {signal}: tb.shape = "
                + str(tb.shape)
                + ", signal.shape = "
                + str(signal_data.shape)
            )

            # At this point, assume that the loaded data is good.
            # Update t_min and append signal and timebase to the working data
            t_min = max(t_min, tb.min())
            t_max = min(t_max, tb.max())
            signal_arrays.append(signal_data)
            tb_arrays.append(tb)

            # TODO: Put this code block into a transform
            # Handle edge-case where the shot is supposedly disruptive, but the disruption
            # happens after the signal ends.
            # In the case where no previously added diagnostic covers the disruption we raise and error
            # If a previously added diagnostic covers the disruption we add dummy data for this signal
            # if self.is_disruptive and self.t_disrupt > tb.max():
            #     t_max_total = tb.max() + signal.get_data_avail_tolerance(self.machine)

            #     if self.t_disrupt > t_max_total:
            #         # The time of the disruption is out of range.
            #         # Instead of the signal we use dummy data.
            #         invalid_signals += 1
            #         tb = np.arange(0, 20.0, 1e-3)
            #         signal = np.zeros((tb.shape[0], signal.shape[1]))
            #         # Setting the entire channel to zero to prevent any peeking into possible disruptions from this early ended channel
            #     else:
            #         t_max = tb.max() + signal.get_data_avail_tolerance(self.machine)
            # else:
            #     t_max = min(t_max, tb.max())

        # Perform sanity checks.
        # 1/ t_max should be larger than t_min
        if t_max < t_min:
            raise BadShotException(
                f"Shot {self.shotnr} has t_max = {t_max} < t_min = {t_min}. Expected t_max > t_min."
            )
        # TODO: Move this test so we don't have to pass conf
        #        # 2/ The shot timebase should be sufficiently long enough.
        #        if (t_max - t_min)/dt <= (2 * conf['model']['length'] + conf['data']['T_min_warn']):
        #            raise BadShotException(f"Shot {self.number} contains insufficient data, omitting.")
        # 3/ The shot is marked disruptive and the disruption occurs after all measurements
        # if self.is_disruptive and self.t_disrupt > t_max:
        #     raise BadShotException(
        #         f"Shot {self.number} is disruptive at {self.t_disrupt}s but data stops at {t_max}"
        #     )

        if invalid_signals > 2:
            raise BadShotException(f"Shot {self.number} has more than 2 bad channels.")

        # if self.is_disruptive:
        #     t_max = self.t_disrupt
        return tb_arrays, signal_arrays, t_min, t_max

    def __len__(self):
        """Number of time samples in the dataset."""
        return len(self.tb)

    def __getitem__(self, idx):
        """Fetches a single sample."""
        return self.signal_tensor[idx, :]
        # return torch.hstack([self.tb_rs.unsqueeze(1), self.signal_tensor[:, idx]])


# end of file frnn_dataset.py
