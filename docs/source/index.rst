.. frnn_loader documentation master file, created by
   sphinx-quickstart on Thu Jul 28 16:46:08 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to frnn_loader's documentation!
=======================================

This package provides a pytorch DataSet interface to the FRNN data. It allows you to
conveniently assemble multi-diagnostic datasets for tokamak discharges and make them
accessible as a `pytorch Dataset <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_


The major components of the package are three classes:

* Signals - 0d, 1d, and 2d measurements from fusion experiments
* Backends - Storage backends that define layout of downloaded data on disk
* Fetchers - Methods to remotely download data from fusion experiments

Here is text :)

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
