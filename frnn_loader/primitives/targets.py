#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from math import ceil
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.utils.errors import BadDataException


"""
    Implements transformation of signal data into targets.

    Targets are synthetic time series that are calculated from shot data.
    This includes 
    * time-to-disruption 
    * time-to-ELM
"""


import logging


class target:
    """target - Abstract base class"""


class target_NULL(target):
    """NullTarget - A dummy class that outputs all zeros."""

    requires_signal = None

    def __call__(self, tb, signal=None):
        return torch.zeros_like(tb)


class target_TTD(target):
    """Time-To-Disruption.

    Transforms a time series into a logarithmic count-down.
    """

    required_signal = None

    def __init__(self, dt, is_disruptive, ttd_max=200.0):
        self.dt = dt
        self.is_disruptive = is_disruptive
        self.ttd_max = ttd_max

    def pick_predictor(self, predictors):
        """Picks a predictor to calculate target from.

        The TTD target is calculated only from the time base. Thus it requires no predictor
        and returns None
        """
        return None

    def __call__(self, tb, signal=None):
        if self.is_disruptive:
            target = max(tb) - tb
            # Maximum time to disruption
            target = np.clip(target, self.ttd_max)
        else:
            target = self.ttd_max * np.ones_like(tb)
            #
        target = np.log10(target + 0.1 * self.dt)
        return target


class target_TTELM(target):
    """Time-To-Elm.
    
    This class implements the time-to-elm target.
    ELMs are identified as peaks in a given signal. The target is a count-down to these peaks.
    

    The signal may look like this. Each peak is an individual ELM:

            |\    |\
            | |   | |
    _______/  |__/  |____

    The constructed target dwells at an offset (here taken to be 10). As the ELM approaches it
    rapidly decreases to zero.
    10: ------\  ----\  ------
    ...        \      \
      0:        |      |

    
    ELMs are identifed by the member function `peak_detection`. With information on the peaks,
    an array countaining the time to each peak is constructed. It defaults to max_ttelm and decreases
    linearly to 0 as an ELM approaches.

    """

    # Signals required to build this target. See data/d3d_signals.yaml
    required_signal = signal_0d("fs07")

    def __init__(
        self,
        max_ttelm=10.0,
        dt=1.0,
        threshold=lambda x: x.mean() * 1.5,
        deadtime=5.0,
        peak_width=5,
    ):
        """
        Args:
            max_ttelm, float: Maximum time to ELM. in milliseconds.
            dt, float: Sample spacing, in milliseconds
            thresold, callable:
            deadtime, float: Separation of peaks, in milliseconds
            peak_width, int: Number of neighbouring elements a peak has to exceed

        """
        self.max_ttelm = max_ttelm
        self.dt = dt
        logging.info(
            f"{self.max_ttelm}, {self.dt}, {type(self.max_ttelm)}, {type(self.dt)}"
        )
        self.max_ttelm_ix = int(self.max_ttelm / self.dt)
        self.threshold = threshold
        self.deadtime = deadtime
        self.deadtime_ix = int(ceil(self.deadtime / dt))
        self.peak_width = int(peak_width)

    def pick_predictor(self, predictors):
        """Picks predictor to calculate TTELM target.

        Time-to-ELM is calculated from FS07.
        """
        return predictors.index(self.required_signal)

    def __call__(self, tb, signal):
        """Returns time-to-ELM, calculated from signal.

        Args:
            tb: torch.tensor: Time-base for a signal that contains ELMs
            signal: torch.tensor: Signal that contains ELMs

        Output:
            ttelm: torch.tensor: Time-base that counts down to time of next ELM
        """
        assert signal.size == signal.size

        # Detect indices where ELM appear.
        elm_idx = self.peak_detection(signal)

        # Sort indices
        elm_idx, _ = elm_idx.sort()
        torch.flip(elm_idx, dims=[0])
        # Initialize ttelm arry with default value
        ttelm = self.max_ttelm * torch.ones_like(tb)

        # Look-back used to fill up ttelm
        lookback = torch.clip(
            torch.hstack([torch.tensor([5]), elm_idx[1:] - elm_idx[:-1]]),
            0,
            self.deadtime_ix,
        )

        for lb, ix in zip(lookback, elm_idx):
            print(lb, ix, ttelm[ix + 1 - lb + 1 : ix + 1])
            # Insert linear count-down windows into ttelm target
            ttelm[ix - lb + 1 : ix + 1] = ttelm[ix - lb + 1] + (
                0.0 - ttelm[ix - lb]
            ) / lb * torch.arange(1, lb + 1)
            print("after: ", ttelm[ix + 1 - lb : ix + 1])

        return ttelm

    def peak_detection(
        self, signal
    ):  # , deadtime=5, threshold=self.threshold, peak_width=5):
        """Detects ELMs in a time series.

        ELMs are defined as peaks in a time seris that exceed a threshold.
        The default threshold is 3 * mean(signal)


        Starting from the largest burst event in the time series at hand, we identify a set of
        disjunct sub records, placed symmetrically around the peak of burst events which exceed
        a given amplitude threshold until no more burst events exceeding this threshold are
        left uncovered.
        Used in Kube et al. PPCF 58, 054001 (2016).
        Input:
        ========
        signal.........ndarray,  float: Timeseries to scan for peaks
        dead_time......integer:  Separation of peaks in sampling points
        threshold......callable: Threshold a peak has to exceeed
        peak_width.....integer:  Number of neighbouring elements a peak has to exceed
        Output:
        ========
        peak_idx_list...ndarray, integer: Indices of peaks in signal
        """

        # Sort time series by magnitude and flip so that index to largest element comes first
        _, max_idx = torch.sort(signal)
        max_idx = torch.flip(max_idx, [0])

        # Remove peaks within dead_time to the array boundary
        max_idx = max_idx[max_idx > self.deadtime_ix]
        max_idx = max_idx[max_idx < signal.size(dim=0) - self.deadtime_ix]

        max_values = np.zeros_like(signal[max_idx])
        max_values[:] = np.squeeze(signal[max_idx])

        # Number of peaks exceeding threshold
        num_big_ones = torch.sum(signal > self.threshold(signal))
        try:
            max_values = max_values[:num_big_ones]
            max_idx = max_idx[:num_big_ones]
        except:
            raise BadDataException(
                "detect_peaks_1d: No peaks in the unmasked part of the array."
            )

        # Mark the indices we need to skip here
        max_idx_copy = torch.zeros_like(max_idx)
        max_idx_copy[:] = max_idx

        # Eliminate values exceeding the threshold within dead_time of another
        # for idx, mv in enumerate(max_values):
        # print 'iterating over %d peaks' % ( np.size(max_idx))
        for i, idx in enumerate(max_idx):
            current_idx = max_idx_copy[i]
            if max_idx_copy[i] == -1:
                #    print 'idx %d is zeroed out' % (idx)
                continue

            # Check if this value is larger than the valueghbouring values of the
            # signal. If it is not, continue with next iteration of for loop
            if (
                signal[current_idx]
                < signal[current_idx - self.peak_width : current_idx + self.peak_width]
            ).any():
                max_idx_copy[i] = -1
                continue

            # Zero out all peaks closer than dead_time
            close_idx = torch.abs(max_idx_copy - idx)
            close_ones = torch.squeeze(torch.where(close_idx < self.deadtime_ix)[0])
            max_idx_copy[close_ones] = -1
            # Copy back current value
            max_idx_copy[i] = max_idx[i]

        # Remove all entries equal to -1
        max_idx_copy = max_idx_copy[max_idx_copy != -1]
        max_idx_copy = max_idx_copy[max_idx_copy < signal.size(dim=0)]

        # Return an ndarray with all peaks of large amplitude indices
        return max_idx_copy
