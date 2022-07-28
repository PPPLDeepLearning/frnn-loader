# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import UnivariateSpline
import logging


from frnn_loader.primitives.signal import Signal
from frnn_loader.utils.errors import SignalCorruptedError


class ProfileSignal(Signal):
    """Represents a profile signal."""

    def __init__(
        self,
        description,
        paths,
        machines,
        tex_label=None,
        causal_shifts=None,
        mapping_range=(0, 1),
        num_channels=32,
        data_avail_tolerances=None,
        is_strictly_positive=False,
        mapping_paths=None,
    ):
        super(ProfileSignal, self).__init__(
            description,
            paths,
            machines,
            tex_label,
            causal_shifts,
            is_ip=False,
            data_avail_tolerances=data_avail_tolerances,
            is_strictly_positive=is_strictly_positive,
            mapping_paths=mapping_paths,
        )
        self.mapping_range = mapping_range
        self.num_channels = num_channels

    def load_data(self, prepath, shot, dtype="float32"):
        """Loads data from txt file and peforms data wrangling into a profile.

        Args:
            prepath:
            shot:
            dtype:

        Returns:
            t: ndarray(float) Signal time base
            sig_inter: Interpolated signal

        Raises
            SignalCorruptedError: When the interval where the current threshold is satisfied is too short.
                                  When the time interval is too short
                                  If the dynamic range of the signal is too low
                                  If the timebase or the signal contains NaNs

        """
        data = self._load_data_from_txt_safe(prepath, shot)

        if np.ndim(data) == 1:
            data = np.expand_dims(data, axis=0)

        # time is stored twice, once for mapping and once for signal
        T = data.shape[0] // 2
        mapping = data[:T, 1:]
        remapping = np.linspace(
            self.mapping_range[0], self.mapping_range[1], self.num_channels
        )
        t = data[:T, 0]
        sig = data[T:, 1:]
        if sig.shape[1] < 2:
            err_msg = f"""Signal {self.description}, shot {shot.number}
should be profile but has only one channel. Possibly only
one profile fit was run for the duration of the shot and 
was transposed during downloading. Need at least 2."""
            logging.error(err_msg)
            raise SignalCorruptedError(err_msg)

        if len(t) <= 1 or (np.max(sig) == 0.0 and np.min(sig) == 0.0):
            err_msg = f"Signal {self.description}, shot {shot.number} contains no data "
            logging.error(err_msg)
            raise SignalCorruptedError(err_msg)

        if np.any(np.isnan(t)) or np.any(np.isnan(sig)):
            err_msg = (
                f"Signal {self.description}, shot {shot.number} contains NaN value(s)"
            )
            logging.error(err_msg)
            raise SignalCorruptedError(err_msg)

        #
        timesteps = len(t)
        sig_interp = np.zeros((timesteps, self.num_channels))
        for i in range(timesteps):
            # make sure the mapping is ordered and unique
            _, order = np.unique(mapping[i, :], return_index=True)
            if sig[i, order].shape[0] > 2:
                # ext = 0 is extrapolation, ext = 3 is boundary value.
                f = UnivariateSpline(mapping[i, order], sig[i, order], s=0, k=1, ext=3)
                sig_interp[i, :] = f(remapping)
            else:
                err_msg = f"""Signal {self.description}, shot {shot.number} 
has insufficient points for linear interpolation. 
dfitpack.error: (m>k) failed for hidden m: fpcurf0:m=1"""
                logging.error(err_msg)
                raise SignalCorruptedError(err_msg)

        return t, sig_interp

    def fetch_data(self, machine, shot_num, c):
        time, data, mapping, success = self._fetch_data_basic(machine, shot_num, c)
        path = self.get_path(machine)
        mapping_path = self.get_mapping_path(machine)

        if mapping is not None and np.ndim(mapping) == 1:
            # make sure there is a mapping for every timestep
            T = len(time)
            mapping = np.tile(mapping, (T, 1)).transpose()
            assert mapping.shape == data.shape, "mapping and data shapes are different"
        if mapping_path is not None:
            # fetch the mapping separately
            (time_map, data_map, mapping_map, success_map) = self.fetch_data_basic(
                machine, shot_num, c, path=mapping_path
            )
            success = success and success_map
            if not success:
                print(
                    "No success for signal {} and mapping {}".format(path, mapping_path)
                )
            else:
                assert np.all(
                    time == time_map
                ), "time for signal {} and mapping {} ".format(
                    path, mapping_path
                ) + "don't align: \n{}\n\n{}\n".format(
                    time, time_map
                )
                mapping = data_map

        if not success:
            return None, None, None, False
        return time, data, mapping, success


# End of file profilesignal.py
