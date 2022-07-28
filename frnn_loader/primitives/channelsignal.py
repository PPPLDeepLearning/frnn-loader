# -*- coding: utf-8 -*-

import re
import numpy as np
from os.path import join
from frnn_loader.primitives.signal import Signal
from frnn_loader.utils.processing import get_individual_shot_file
from frnn_loader.utils.errors import DataFormatError

class ChannelSignal(Signal):
    """A signal best represented by a number of channels."""
    def __init__(self, description, paths, machines, tex_label=None,
                 causal_shifts=None, data_avail_tolerances=None,
                 is_strictly_positive=False, mapping_paths=None):
        """Initializes the ChannelSignal

        Args:
            description (string) :
            paths (list[string]) :
            machines (list :obj:`plasma.primitives.machine.Machine) :
            tex_label (str) :
            causal_shifts :
            data_avail_tolerances :
            is_strictly_positive :
            mapping_paths :

        """
        super(ChannelSignal, self).__init__(
            description, paths, machines, tex_label, causal_shifts,
            is_ip=False, data_avail_tolerances=data_avail_tolerances,
            is_strictly_positive=is_strictly_positive,
            mapping_paths=mapping_paths)
        nums, new_paths = self.get_channel_nums(paths)
        self.channel_nums = nums
        self.paths = new_paths

    def get_channel_nums(self, paths):
        """A mystery function."""
        regex = re.compile(r'channel\d+')
        regex_int = re.compile(r'\d+')
        nums = []
        new_paths = []
        for p in paths:
            assert(p[-1] != '/')
            elements = p.split('/')
            res = regex.findall(elements[-1])
            assert(len(res) < 2)
            if len(res) == 0:
                nums.append(None)
                new_paths.append(p)
            else:
                nums.append(int(regex_int.findall(res[0])[0]))
                new_paths.append("/".join(elements[:-1]))

        return nums, new_paths

    def get_channel_num(self, machine):
        idx = self.get_idx(machine)
        return self.channel_nums[idx]

    def fetch_data(self, machine, shot_num, c):
        """Fetches signal data.

            Args:
                machine (:obj:`plasma.primitives.machine.Machine`)
                shot_num (int) :
                c (???) :

            Returns
                Nothing

            Raises:
                DataFormatError : When the loaded data array has less than 2 dimensions.

        """
        time, data, mapping = self.fetch_data_basic(machine, shot_num, c)
        if np.ndim(data) != 2:
            raise DataFormatError(f"Channel Signal {self} expected 2D array for shot {self.shotnumber}")

        mapping = None  # we are not interested in the whole profile
        channel_num = self.get_channel_num(machine)
        data = data[channel_num, :]  # extract channel of interest

        return time, data, mapping

    def get_file_path(self, prepath, machine, shot_number):
        dirname = self.get_path(machine)
        dirname = join(dirname, f"channel{self.get_channel_num(machine)}")
        return get_individual_shot_file(join(prepath, machine.name, dirname), shot_number)

# end of file channelsignal.py
