# -*- coding: utf-8 -*-

"""Backends define mappings between remote data and locally cached version.

Measurements from fusion experiments are typically accessed through `MDSplus <https://www.mdsplus.org>`_
Backends define a mapping from MDSplus to local files. How individual signals are mapped to
files is defined in the yaml specification provided `here` <https://github.com/PlasmaControl/d3d_signals/>

As an example, consider

.. code-block::

    q95:
        Machine: D3D
        Description: Q95 safety factor
        MDSTree: EFIT01
        MDSPath: RESULTS.AEQDSK:Q95
        LocalPath: q95
        ndim: 0
        Channels: 1

This block defines a signal called ``q95``. The mapping from MDSplus signal to file is
implemented through the mappings ``MDSTree``, ``MDSPath``, and ``LocalPath``. Together,
the two MDS keys allow to construct the location in MDSplus of the signal. Then
``LocalPath`` is used to define a location in file.

All backends implemend a ``load`` and a ``store`` method. 
The basic usage for any ``load`` routine is

.. code-block::

  shotnr = 123456
  signal = signal_0d("q95")
  my_backend = backend_txt("/path/to/data")
  tb, data = my_backend.load(signal.info, shotnr)


The basic usage for any ``store`` routine is

.. code-block::

    shotnr = 123456
    signal = signal_0d("q95")
    my_fetcher = fetcher_d3d()
    my_backend = backend_txt("/path/to/data")
    # Code that fetches the data
    xdata, _, zdata, _, _, _ = my_fetcher.fetch(signal.info, shotnr)
    # Store using the backend
    my_backendstore(signal.info, shotnr, xdata, zdata)


In the case of the ``q95`` data, the base directory ``/path/to/data``
would be empty before the call to ``fetch``.




Compared to MDSplus, backends only store limited data. From D3D for example,
MDSplus provides a timebase for any signal. In the case of two-dimensinoal data,
MDSplus also provides coordinates for the second dimension. In addition, MDSplus
allows to query the dimension of the measurements. Backends neglect all dimensions.
It is further assumed that all time is in units of milliseconds.

"""


from os.path import join
import torch


class backend:
    """Abstract basis class for all backends.

    Args:
        root (string) : Root path of the data directory
        dtype (torch.dtype, optional) : Datatype of the return tensor
    """

    def __init__(self, root, dtype=torch.float32):
        self.root = root
        self.dtype = dtype

    def _mapping_path(self, sig_info, shotnr):
        """Constructs a path to load/store the file from.

        >>> from frnn_loader.primitives import signal_0d
        >>> from frnn_loader.backends.backend_txt import backend_txt
        >>> signal_fs07 = signal_0d("fs07")
        >>> my_backend = backend_txt("/home/rkube/datasets/frnn/dataset01")
        >>> my_backend._construct_file_path(signal_fs07.info, 180400)
            /home/rkube/datasets/frnn/dataset01/d3d/fs07/180400.txt

        This routine accesses dictionary keys ``Machine`` and ``LocalPath``
        defined through `d3d_signals.yaml`


        Args
            sig_info (dict): Dictionary describing a user signal
            shotnr (int): Shot number

        Returns
            string: File path to the data file.
        """
        return join(self.root, sig_info["Machine"], sig_info["LocalPath"])


# end of file backend.txt
