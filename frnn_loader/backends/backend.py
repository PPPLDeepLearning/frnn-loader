# -*- coding: utf-8 -*-

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

    def _construct_file_path(self, sig_info, shotnr):
        """Constructs a path to load/store the file from.

        >>> from frnn_loader.primitives import signal_0d
        >>> from frnn_loader.backends.backend_txt import backend_txt
        >>> signal_fs07 = signal_0d("fs07")
        >>> my_backend = backend_txt("/home/rkube/datasets/frnn/dataset01")
        >>> my_backend._construct_file_path(signa_fs07.info, 180400)
            /home/rkube/datasets/frnn/dataset01/d3d/fs07/180400.txt

        This routine uses dictionary keys

        Args:
            sig_info (dict): Dictionary describing a user signal
            shotnr (int): Shot number

        Returns:
            string: File path to the data file.
        """
        return join( self.root, sig_info["Machine"], sig_info["LocalPath"])


# end of file backend.txt
