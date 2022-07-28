# -*- coding: utf-8 -*-
"""Definitions of machine backends.

This module contains the machine class and derived classes. 
These classes define methods for downloading the respective signals.

Derived classes include D3D, JET, and NSTX.
"""

import numpy as np
import logging

from frnn_loader.utils.errors import SignalNotFoundError
from frnn_loader.backends.machine_helper import (
    create_missing_value_filler,
    get_tree_and_tag,
    get_tree_and_tag_no_backslash,
)


class Machine:
    """Abstraction of a machine

    This class collects information about
    * D3D
    * JET
    * NSTX

    and provides abstractions for fetching the data
    """

    def __init__(self, current_threshold=1e-8):
        """Initializes Machine object

        Args:
            current_threshold: Minimum value of the plasma current that defines the active
                               shot phase for this machine.

        Attributes:
            current_threshold: Minimum value of the plasma current that defines the active
                               shot phase for this machine.

        """
        assert current_threshold > 0.0

        self.current_threshold = current_threshold

    def __eq__(self, other):
        """Equality is defined by matching names"""
        return self.name.__eq__(other.name)

    def __lt__(self, other):
        return self.name.__lt__(other.name)

    def __ne__(self, other):
        return self.name.__ne__(other.name)

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def fetch_data(self):
        """Overridden by derived classes."""
        raise NotImplementedError(
            "Machine.fetch_data should be implemented by derived classes."
        )


class MachineNSTX(Machine):
    # Name and Server are class attributes
    name = "NSTX"
    server = "skylark.pppl.gov:8501::"

    def __init__(self, current_threshold=2e-1):
        super(MachineNSTX, self).__init__(current_threshold)
        self.name = "NSTX"

    def fetch_data(signal_path, shot_num, c):
        """Fetch NSTX data

        Args:
          signal_path: Path to signals?
          shot_num: Shot number (integer)
          c: ???

        Returns:
          time: Time base for the desired signal
          data: Data of requested signal
        """
        assert shot_num > 0

        tree, tag = get_tree_and_tag(signal_path)
        c.openTree(tree, shot_num)
        data = c.get(tag).data()
        time = c.get("dim_of(" + tag + ")").data()

        return time, data


class MachineJET(Machine):
    name = "jet"
    server = "mdsplus.jet.efda.org"

    """JET Machine.

    Attributes:
        name (str): "JET"
        server (str): "mdsplus.jet.efda.org"

    """

    def __init__(self, current_threshold=1e5):
        super(MachineJET, self).__init__(current_threshold)

    def fetch_data(self, signal_path, shot_num, c):
        """Fetch JET data

        Args:
          signal_path: Path to signals?
          shot_num: Shot number (integer)
          c: ???

        Returns:
          time: Time base for the desired signal
          data: Data of requested signal
          ydata: True
        """
        time = np.array([0])
        ydata = None
        data = np.array([0])

        data = c.get(f'_sig=jet("{signal_path}/",{shot_num})').data()
        if np.ndim(data) == 2:
            data = np.transpose(data)
            time = c.get(f'_sig=dim_of(jet("{signal_path}/",{shot_num}),1)').data()
            ydata = c.get(f'_sig=dim_of(jet("{signal_path}/",{shot_num}),0)').data()
        else:
            time = c.get('_sig=dim_of(jet("{signal_pat}/",{shot_num}))').data()

        return time, data, ydata


class MachineD3D(Machine):
    # Attributes are static
    name = "d3d"
    server = "atlas.gat.com"
    """D3D Machine."""

    def __init__(self, current_threshold=2e-1):
        super(MachineD3D, self).__init__(current_threshold)

    def fetch_data(MachineD3D, signal_path, shot_num, c):
        """Fetch D3D data

        Args:
          signal_path: Path to signals?
          shot_num: Shot number (integer)
          c: ???

        Returns:
          time: Time base for the desired signal
          data: Data of requested signal
          None: Legact
          found: True
        """

        tree, signal = get_tree_and_tag_no_backslash(signal_path)
        if tree is None:
            signal = c.get('findsig("' + signal + '",_fstree)').value
            tree = c.get("_fstree").value

        # Retrieve data
        found = False
        xdata = np.array([0])
        ydata = None
        data = np.array([0])

        # Retrieve data from MDSplus (thin)
        # first try, retrieve directly from tree andsignal
        def get_units(str):
            units = c.get("units_of(" + str + ")").data()
            if units in ["", " "]:
                units = c.get("units(" + str + ")").data()
            return units

        try:
            c.openTree(tree, shot_num)
            data = c.get("_s = " + signal).data()
            # data_units = c.get('units_of(_s)').data()
            rank = np.ndim(data)
            found = True

        except Exception as e:
            logging.error(e)
            raise (e)
            pass

        # Retrieve data from PTDATA if node not found
        if not found:
            # g.print_unique("not in full path {}".format(signal))
            data = c.get('_s = ptdata2("' + signal + '",' + str(shot_num) + ")").data()
            if len(data) != 1:
                rank = np.ndim(data)
                found = True

        # Retrieve data from Pseudo-pointname if not in ptdata
        if not found:
            # g.print_unique("not in PTDATA {}".format(signal))
            data = c.get('_s = pseudo("' + signal + '",' + str(shot_num) + ")").data()
            if len(data) != 1:
                rank = np.ndim(data)
                found = True
        # this means the signal wasn't found
        if not found:
            raise SignalNotFoundError(f"No such signal: {signal}")

        # get time base
        if rank > 1:
            xdata = c.get("dim_of(_s,1)").data()
            # xunits = get_units('dim_of(_s,1)')
            ydata = c.get("dim_of(_s)").data()
            # yunits = get_units('dim_of(_s)')
        else:
            xdata = c.get("dim_of(_s)").data()
            # xunits = get_units('dim_of(_s)')

        # MDSplus seems to return 2-D arrays transposed.  Change them back.
        if np.ndim(data) == 2:
            data = np.transpose(data)
        if np.ndim(ydata) == 2:
            ydata = np.transpose(ydata)
        if np.ndim(xdata) == 2:
            xdata = np.transpose(xdata)

        xdata = xdata * 1e-3  # time is measued in ms
        return xdata, data, ydata


# End of file machine.py
