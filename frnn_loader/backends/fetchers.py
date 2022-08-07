# -*- coding: UTF-8 -*-
import time
import numpy as np
import MDSplus as mds
import yaml
from frnn_loader.utils.errors import BadDownloadError, MDSNotFoundException

def mds_get_units(string, c):
    """Returns units of an MDS expression"""
    units = c.get(f'units_of({string})').data()
    if units == '' or units == ' ':
        units = c.get(f'units({string})').data()
    return units

class fetcher():
    def __init__(self):
        pass

class fetcher_d3d_v1():
    """Fetch data from D3D using MDSplus.


    This file is heavily influenced by GA's gadata class.
    Main modification are
    * Porting to python3, including f-strings
    * Removing all error handling.
    * Removing the True flag

    Args:
        mds_hhostname (str): Hostname of the MDS server


    """   

    def __init__(self, mds_hostname="atlas.gat.com"):
        # Connect to D3D MDplus server
        self.mds_hostname = mds_hostname
        self.conn = mds.Connection(mds_hostname)


    def fetch(self, signal_info, shotnr):
        """Fetch data from D3D MDS server
        
        Args:
            signal_info (dict): Info dictionary from shot class
            shotnr (int): Shot number

        Returns:
            xdata (ndarray)
            data  (ndarray)
            ydata (ndarray)

        Raises:
            BadDownloadError - If somehow all data is bad.
        
        """
        t0 = time.time()
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
                xdata = self.conn.get('dim_of(_s)').data()[:]

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
            raise BadDownloadError(f"Data that was downloaded from {self.mds_hostname} is empty")

        return xdata, ydata, zdata, xunits, yunits, zunits

# end of file fetchers.py
