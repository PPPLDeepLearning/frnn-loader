# -*- coding: utf-8 -*-

"""Creates dummy data for use in unit testing."""

import logging
import torch

from frnn_loader.backends.backend import backend

class backend_dummy(backend):
    """Backend that provides dummy data
    
    Args
        root - ignored
        dtype (torch.dtype, default to torch.float32) : Datatype to use.
    """

    def __init__(self, root, dtype=torch.float32):
        super().__init__(root, dtype)

    def load(self, sig_info, shotnr):
        """Loads data.
        
        Args
            sig_info (string): Signal info
            shotnr (int): shot number    
        """

        tb = torch.arange(0.0, 1000.0, 1.0)
        data = torch.arange(0.0, 1000.0, 1.0).unsqueeze(1)

        return tb, data

    def store(self, sig_info, shotnr, tb, signal_data):
        """Store a signal.
        
        This will do nothing"""
        pass