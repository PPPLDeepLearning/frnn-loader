# coding: utf-8 -*-
"""Resamplers are used to bring signal data onto a common time-base.

Most diagnostics sample measurements on their own time-base. To
make a collection of signals available for deep neural network training
we have to bring them to a common time-base.
"""

import torch


class resampler:
    """Abstract basis class.

    Args:
        t_min (float) : Starting time of the resampling time-base
        t_max (float) : End time of the resampling time-base
        dt (float) : Sample spacing

    """

    def __init__(self, t_min, t_max, dt):
        self.t_min = t_min
        self.t_max = t_max
        self.dt = dt

    def __call__(self, tb, sig):
        """Resample a signaal.

        Args:
            tb (torch.tensor) : Time-base of input signal
            sig (torch.tensor) : Samples of the input signal

        Returns:
            tb_new (torch.tensor) : New time bsae
            sig_new (torch.tensor) : Re-sampled signal
        """
        sort_idx = torch.argsort(tb)
        tb = tb[sort_idx]
        sig = sig[sort_idx, :]
        num_channels = sig.shape[1]
        # Allocate tensors for new time base and re-sampled signal
        tb_new = torch.arange(self.t_min, self.t_max, self.dt, dtype=tb.dtype)
        sig_new = torch.zeros((len(tb_new), num_channels), dtype=sig.dtype)

        for i in range(num_channels):
            sig_new[:, i] = self._interp(sig[:, i], tb, tb_new)

        if torch.any(torch.isnan(sig_new)).item():
            raise ValueError("Resampled signal contains NaN")
        if torch.any(torch.isinf(sig_new)).item():
            raise ValueError("Resampled signal contains Inf")

        return tb_new, sig_new

    def __len__(self):
        return int((self.t_max - self.t_min) / self.dt)


class resampler_last(resampler):
    """Uses last previous sample.

    Given a signal sampled with sampling rate dt_old, we wish to re-sample
    it with a sampling rate dt_new.

    For any time t_new, the value of the re-sampled signal is given by the
    u_old(t_old*) such that t_old* = max(t_old) with t_old < t_new.

    I.e we take the last sample value before t_new.

    Usually we wish to interpolate from a slow to a fast time-scale.
    In this situation we often have the case where a signal value is
    repeated multiple times.

    """

    def _interp(self, sig_old, tb_old, tb_new):
        # This is a pure conversion from numpy to torch library calls
        # of the original function `time_sensitive_interp`.

        idx = torch.maximum(
            torch.tensor([0]), torch.searchsorted(tb_old, tb_new, right=True) - 1
        )
        return sig_old[idx]


# end of file resamplers.py
