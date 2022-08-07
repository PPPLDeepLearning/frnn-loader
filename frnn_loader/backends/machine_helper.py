# -*- coding: utf-8 -*-
"""Helper functions for machine classes.

Various helper functions that modify strings for paths etc.
"""

import numpy as np


def create_missing_value_filler():
    """Common filler for missing values.

    Missing values are to be filled with a time-range and corresponding values of zero.
    Time range is 0...100
    Values are 0
    """
    time = np.linspace(0, 100, 100)
    vals = np.zeros_like(time)
    return time, vals


def get_tree_and_tag(path):
    """Fetch tree and tag from a path. No idea how this works."""
    spl = path.split("/")
    tree = spl[0]
    tag = "\\" + spl[1]
    return tree, tag


# End of file machine_helper.py
