# -*- coding: utf-8 -*-

import torch
import numpy as np
from frnn_loader.utils.errors import SignalCorruptedError

"""Construct list of valid shot times.

Measurement data is stored for a time interval around experiments. A plasma shot
may be executed at t=0-7s but measurements would be taken from t=-1.0 - 10.0s.

Ideally we want to crimp the times for a plasma shot.

This file contains a collection of filters that extract the start
and end in a shot when a certain condition is fulfilled.
"""


class filter_ip_thresh:
    """A filter based on plasma current"""

    def __init__(self, ip_thresh=0.2, num_pts=100, abs=True):
        """Initializes Ip threshold filter

        Args:
            ip_thresh (Float): Threshold under which we discard the shot.
            num_pts (int): Number of points that have to exceed the threshold
            abs (bool): If True, test the criterion against the absolute value of the threshold.
                        I.e. account for clock- and anti-clockwise current. Default: True


        """
        self.ip_thresh = ip_thresh
        self.num_pts = num_pts
        self.abs = abs

    def __call__(self, tb, data):
        """Returns min and max time when plasma current is above threshold.

        Args:
            data (array-like) : Plasma current time series
            tb (array-like) : Time-base for plasma current

        Returns:
            tmin, tmax: Interval boundaries where plasma current exceeds the threshold.

        Raises:
            SignalCorruptedError: When fewer than 100 datapoints exceed the filter threshold

        """
        if self.abs:
            if isinstance(data, np.ndarray):
                data = np.abs(data)
            elif isinstance(data, torch.Tensor):
                data = data.abs()
            else:
                data = abs(data)

        good_idx = (data > self.ip_thresh).squeeze()
        if good_idx.sum() < self.num_pts:
            raise SignalCorruptedError(f"{self}: Bad signal: Fewer than {self.num_pts} points are valid.")
        tb_good = tb[good_idx]
        tmin = tb_good.min().item()
        tmax = tb_good.max().item()

        return tmin, tmax

    def __str__(self):
        return f"filter_ip_thresh - self.ip_thresh={self.ip_thresh}"


# end of file filters.py
