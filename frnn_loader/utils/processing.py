'''
#########################################################
This file containts classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

from __future__ import print_function
import itertools

import numpy as np

from os.path import join
# from scipy.interpolate import UnivariateSpline

# interpolate in a way that doesn't use future information.
# It simply finds the latest time point in the original array
# that is less than or equal than the time point in question
# and interpolates to there.



class tb_resampler():
    """Resamples a signal on a time-base"""

    def __init__(self, t_min, t_max, dt):
        """Initializes signal resampler

        Paramters:
            t_min (float): Start time of resampled signal
            t_max (float): Desired end time of resampled signal
            dt (float): Sampling time of resampled signal
        """
        self.t_min = t_min
        self.t_max = t_max
        self.dt = dt

    def resample(self, signal, tb):
        """Resample signal on new time-base

        Parameters:
            signal (ndarray) : Signal that we want to resample
            tb (ndarray) : time-bsae of the passed signal
        """
        # Assert that signal and time base have the same number of elements
        assert(tb.shape[0] == signal.shape[0])




def time_sensitive_interp(x, t, t_new):
    indices = np.maximum(0, np.searchsorted(t, t_new, side='right') - 1)
    return x[indices]


def resample_signal(t, sig, tmin, tmax, dt, dtype=np.float32):
    """Resample a signal onto a new time-base.

    Parameters:
        tb (ndarray, float) : Time-base of the input signal
        sig (ndarray, float) : Input signal
        tmin (float) : Start time of target time-bsae
        tmax (float) : End time of target time-base
        dt (float) : time-step of target time-base

    Returns
        tt (ndarray, float) : Re-sampled time bsae
        sig_rs (ndarray, float) : Re-sampled signal

    Raises:
        ValueError : When the re-sampled signal contains a NaN


    """
    # Find the indices that sort the time bsae
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    sig = sig[sort_idx, :]
    num_channels = sig.shape[1]
    # Allocate arrays for new time-base and re-sampled signal
    tt = np.arange(tmin, tmax, dt, dtype=dtype)
    sig_rs = np.zeros((len(tt), num_channels), dtype=dtype)

    for i in range(num_channels):
        # make sure to not use future information
        sig_rs[:, i] = time_sensitive_interp(sig[:, i], t, tt)
        # f = UnivariateSpline(t,sig[:,i],s=0,k=1,ext=0)
        # sig_interp[:,i] = f(tt)

    if(np.any(np.isnan(sig_rs))):
        raise ValueError("Resampled signal contains NaN")

    return tt, sig_rs


def cut_signal(t, sig, tmin, tmax):
    """Truncate signal to tmin/tmax.

    Args:
        t (array, float) : time base of a signal
        sig (array, float) : signal
        tmin (float): Lower time limit
        tmax (float): Upper time limit

    Returns:
        t (array, float) : timebase truncated to [tmin:tmax]
        sig (array, float) : signal truncated to [tmin, tmax]
    """
    raise DeprecationWarning("This should not be a separate function...")
    #mask = np.logical_and(t >= tmin, t <= tmax)
    #return t[mask], sig[mask, :]


# def cut_and_resample_signal(t, sig, tmin, tmax, dt, dtype=np.float32):
#     t, sig = cut_signal(t, sig, tmin, tmax)
#     return resample_signal(t, sig, tmin, tmax, dt, dtype)


def get_individual_shot_file(prepath, shot_num, ext='txt'):
    return join(prepath, f"{shot_num}.{ext}")


def append_to_filename(path, to_append):
    ending_idx = path.rfind('.')
    new_path = path[:ending_idx] + to_append + path[ending_idx:]
    return new_path


def train_test_split(x, frac, do_shuffle=False):
    if not isinstance(x, np.ndarray):
        return train_test_split_robust(x, frac, do_shuffle)
    mask = np.array(range(len(x))) < frac*len(x)
    if do_shuffle:
        np.random.shuffle(mask)
    return x[mask], x[~mask]


def train_test_split_robust(x, frac, do_shuffle=False):
    mask = np.array(range(len(x))) < frac*len(x)
    if do_shuffle:
        np.random.shuffle(mask)
    train = []
    test = []
    for (i, _x) in enumerate(x):
        if mask[i]:
            train.append(_x)
        else:
            test.append(_x)
    return train, test


def train_test_split_all(x, frac, do_shuffle=True):
    groups = []
    length = len(x[0])
    mask = np.array(range(length)) < frac*length
    if do_shuffle:
        np.random.shuffle(mask)
    for item in x:
        groups.append((item[mask], item[~mask]))
    return groups


def concatenate_sublists(superlist):
    return list(itertools.chain.from_iterable(superlist))


def get_signal_slices(signals_superlist):
    indices_superlist = []
    signals_so_far = 0
    for sublist in signals_superlist:
        indices_sublist = signals_so_far + np.array(range(len(sublist)))
        signals_so_far += len(sublist)
        indices_superlist.append(indices_sublist)
    return indices_superlist
