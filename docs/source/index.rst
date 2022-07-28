.. frnn_loader documentation master file, created by
   sphinx-quickstart on Thu Jul 28 16:46:08 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to frnn_loader's documentation!
=======================================

This package provides a pytorch DataSet interface to the FRNN data.
It consists primarily of classes for 

* Signals - 0d, 1d, and 2d measurements from fusion experiments
* Shots - Collections of Signals from a shot at a machine
* Machine - Abstractions of Devices such as DIII-D or NSTX
* Storage - Storage backends that define layout of downloaded data on disk
* Loading - Methods to remotely download data from fusion experiments


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
