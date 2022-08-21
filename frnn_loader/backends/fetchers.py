# -*- coding: UTF-8 -*-

"""Defines access to signal data using remote data server.

Data signals authorities are the respective experiments MDS servers.
This module deefines classes that allows to fetch signals from these servers.

Signals, together with metadata are defined in frnn_loader/data/d3d_signals.yaml.
In particular, the signal_info dict is constructed using this metadata.
"""

import torch
import MDSplus as mds
import logging
from frnn_loader.utils.errors import BadDownloadError, MDSNotFoundException


class fetcher:
    """Abstract basis class for fetchers."""

    def __init__(self):
        pass

    def fetch(self, signal_info, shotnr):
        pass


class fetcher_d3d_v1:
    """Fetch data from D3D using MDSplus.


    This file is heavily influenced by GA's gadata class.
    Main modification are
    * Porting to python3, including f-strings
    * Removing all error handling.
    * Removing the True flag

    Args:
        mds_hhostname (str): Hostname of the MDS server

    Raises:
        BadDownloadError


    """

    def __init__(self, mds_hostname="atlas.gat.com", dtype=torch.float32):
        # Connect to D3D MDplus server
        self.mds_hostname = mds_hostname
        self.conn = mds.Connection(mds_hostname)
        self.dtype = dtype

    def fetch(self, signal_info, shotnr):
        """Fetch data from D3D MDS server

        Args:
            signal_info (dict): Info dictionary from shot class
            shotnr (int): Shot number

        Returns:
            xdata (torch.tensor) - Time base of the requested data
            ydata (torch.tensor) - Optional ydata returned by MDS
            zdata (torch.tensor) - MDS signal as a 2-dimensional tensor. dim0: Sample. dim1: Channels.
            xunits (string) - (Optional) units of the time base. Can be empty.
            yunits (string) - (Optional) units of the ydata. Can be empty.
            zunits (string) - (Optional) units of the signal. Can be empty.


        Raises:
            BadDownloadError - If the downloaded data contains less than 10 elements.
            RuntimeWarning - If any downloaded data contains Inf or NaN

        """
        # The signal.info dictionary tells us how to load the data from MDS.
        xdata, ydata, zdata = None, None, None
        xunits, yunits, zunits = None, None, None

        # If we have an MDSTree and MDSPath we use them to load the data. The code below is
        # basically gadata.py
        if "MDSTree" in signal_info.keys():
            self.conn.openTree(signal_info["MDSTree"], shotnr)
            zdata = self.conn.get(f"_s = {signal_info['MDSPath']}").data()
            zunits = self.conn.get("units_of(_s)").data()
            xdata = self.conn.get("dim_of(_s)").data()
            xunits = self.conn.get("units_of(dim_of(_s))").data()

            if zdata.ndim > 1:
                print("MDS ")
                ydata = self.conn.get("dim_of(_s, 1)").data()
                yunits = self.conn.get("units_of(dim_of(_s, 1)").data()
                if ydata.ndim == 2:
                    ydata = ydata.T

            if zdata.ndim == 2:
                zdata = zdata.T

            if xdata.ndim == 2:
                xdata = xdata.T

        # If we have a PTdata in the keys, assume we need to fetch the data using the ptdata2 call:
        elif "PTData" in signal_info.keys():
            s = signal_info["PTData"]
            zdata = self.conn.get(f'_s = ptdata2("{s}", {shotnr})').data()[:]
            if len(zdata) != 1:
                xdata = self.conn.get("dim_of(_s)").data()[:]

        # Hack: If we can get a size, assume a return element i a numpy array.
        # IF any of them is has positive size think that we got good data back.
        bad_size = True
        for d in [xdata, ydata, zdata, xunits, yunits, zunits]:
            try:
                if d.size > 10:
                    bad_size = False
                    break
            except:
                continue
        if bad_size:
            raise BadDownloadError(
                f"Data that was downloaded from {self.mds_hostname} is empty"
            )

        if xdata is not None:
            xdata = torch.tensor(xdata, dtype=self.dtype)
            if torch.any(torch.isinf(xdata)).item():
                raise RuntimeWarning("zdata contains inf values")

            if torch.any(torch.isnan(xdata)).item():
                raise RuntimeWarning("zdata contains NaN values")

        if ydata is not None:
            ydata = torch.tensor(ydata, dtype=self.dtype)
            if torch.any(torch.isinf(ydata)).item():
                raise RuntimeWarning("ydata contains inf values")

            if torch.any(torch.isnan(ydata)).item():
                raise RuntimeWarning("ydata contains NaN values")

        if zdata is not None:
            zdata = torch.tensor(zdata, dtype=self.dtype)
            if torch.any(torch.isinf(zdata)).item():
                raise RuntimeWarning("xdata contains inf values")

            if torch.any(torch.isnan(zdata)).item():
                raise RuntimeWarning("xdata contains NaN values")

            if signal_info["ndim"] == 0:
                zdata = zdata.unsqueeze(1)
                # logging.debug(f"Downloaded 0d signal: {signal_info}. zdata.shape = {zdata.shape}")

        # Final asserts on the shape of the array.
        # zdata shold be two-dimensional. Dim0: Sample. Dim1: Channel (this is [nsample, 1] for 0d data.)
        assert zdata.ndim == 2

        return xdata, ydata, zdata, xunits, yunits, zunits


# end of file fetchers.py
