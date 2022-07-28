#!/usr/bin/python
#-*- coding: UTF-8 -*-

#import os
#import subprocess
from setuptools import setup, find_packages


setup(name="frnn_loader",
      version="0.0.1",
      packages=find_packages(),
      # scripts = [""],
      description="Pytorch-style DataLoaders for FRNN dataset.",
      long_description="""Add description here""",
      author="Ralph Kube",
      author_email="rkube@pppl.gov",
      maintainer="Ralph Kube",
      maintainer_email="rkube@pppl.gov",
      # url = "http://",
      download_url="https://git.pppl.gov/rkube/frnn-loader",
      # license = "Apache Software License v2",
      test_suite="tests",
      install_requires=[
          'torch>=1.12',
          'scipy',
          'numpy>=1.23',
          'sphinx',
          'sphinx-rtd-theme'
          ],
      # TODO(KGF): add optional feature specs for [deephyper,balsam,
      # readthedocs,onnx,keras2onnx]
      tests_require=[],
      classifiers=["Development Status :: 3 - Alpha",
                   "Environment :: Console",
                   "Intended Audience :: Science/Research",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering :: Information Analysis",
                   "Topic :: Scientific/Engineering :: Physics",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: System :: Distributed Computing",
                   ],
      platforms="Linux",
      )
