# coding: utf-8 -*-
"""Resamplers are used to bring signal data onto a common time-base.

Diagnostics sample measurements on their own time-base. To
make a collection of signals available for deep neural network training,
they need to be re-sampled onto a common time-base.
"""

import torch


class resampler:
    """Abstract basis class.

    Args:
        t_start (float) : Starting time of the resampling time-base
        t_end (float) : End time of the resampling time-base
        dt (float) : Sample spacing

    """

    def __init__(self, t_start, t_end, dt, dtype=torch.float32):
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.dtype = dtype

    def __call__(self, tb, sig):
        """Resample a signal.

        The data type of the new time-base and the resampled signal is defined by self.dtype

        Args:
            tb (torch.tensor) : Time-base of input signal
            sig (torch.tensor) : Samples of the input signal

        Returns:
            tb_new (torch.tensor) : New time bsae
            sig_new (torch.tensor) : Re-sampled signal
        """
        assert tb.dtype == sig.dtype == self.dtype
        sort_idx = torch.argsort(tb)
        tb = tb[sort_idx]
        sig = sig[sort_idx, :]
        num_channels = sig.shape[1]
        # Allocate tensors for new time base and re-sampled signal
        tb_new = torch.arange(self.t_start, self.t_end, self.dt, dtype=self.dtype)
        sig_new = torch.zeros((len(tb_new), num_channels), dtype=self.dtype)

        for i in range(num_channels):
            sig_new[:, i] = self._interp(sig[:, i], tb, tb_new)

        if torch.any(torch.isnan(sig_new)).item():
            raise ValueError("Resampled signal contains NaN")
        if torch.any(torch.isinf(sig_new)).item():
            raise ValueError("Resampled signal contains Inf")

        return tb_new, sig_new

    def __len__(self):
        return int((self.t_end - self.t_start) / self.dt)


class resampler_causal(resampler):
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
