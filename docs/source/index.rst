.. frnn_loader documentation master file, created by
   sphinx-quickstart on Thu Jul 28 16:46:08 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to frnn_loader's documentation!
=======================================

This package provides a pytorch DataSet interface to the FRNN data. It allows to
conveniently assemble multi-diagnostic datasets for tokamak discharges and make them
accessible as a `pytorch Dataset <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_

Here is a basic example to get started:

.. code-block:: 
    
      from frnn_loader.primitives.resamplers import resampler_causal
      from frnn_loader.backends.backend_txt import backend_txt
      from frnn_loader.backends.fetchers import fetcher_d3d_v1
      from frnn_loader.primitives.signal import signal_0d
      from frnn_loader.loaders.frnn_dataset import shot_dataset

      # Re-sample the signal onto the interval [0.0:2.0] seconds, using a time-step of 1ms:
      my_resampler = resampler_causal(0.0, 2.0, 1e-3)

      # Instantiate a file backend. This implies that all signals are stored in the 
      # path pointed to by root:
      root = "/home/rkube/datasets/frnn/signal_data_new_2021/"
      my_backend_file = backend_txt(root)
      my_fetcher = fetcher_d3d_v1()

      # Define the signals
      signal_list = [signal(n) for n in  ["dens", "fs07", "q95"]]

      # Instantiate a dataset
      ds = shot_dataset(184800, [signal_fs07], 
                        resampler=my_resampler, 
                        backend_file=my_backend_file, 
                        backend_fetcher=my_fetcher,
                        download=True, 
                        dtype=torch.float32)

      # We can now iterate over the datasets samples:
      for item in ds:
            # Do things

The major components of the package are three classes:

* Signals - 0d, 1d, and 2d measurements from fusion experiments
* Backends - Storage backends that define layout of downloaded data on disk
* Fetchers - Methods to remotely download data from fusion experiments

:class:`frnn_loader.shot_dataset` ties these components together in a way that is compatible
with pytorch datasets.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   deployment
   modules




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
