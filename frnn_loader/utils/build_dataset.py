# -*- coding: utf-8 -*-

import sys

sys.path.append("/home/rkube/repos/frnn-loader")

from os.path import join
from pathlib import Path

import yaml

import torch

from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.backends.backend_hdf5 import backend_hdf5
from frnn_loader.primitives.filters import filter_ip_thresh
from frnn_loader.primitives.resamplers import resampler_causal
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.primitives.normalizers import mean_std_normalizer
from frnn_loader.loaders.frnn_dataset_disk import shot_dataset_disk


"""Construct a dataset for FRNN training.

Predictive machine learning models are trained on datasets. These dataset
consist of a suite of measurements taken on a set of shots.

Deep neural networks are trained on pre-processed and normalized data.
Pre-processing includes:
- Resampling of the measurements onto a common time-base
- Construction of target variables, such as time-to-disruption or time-to-ELM
- Signal clipping

Normalization means the transformation of signals into order unity quantities. Common ways
to do this is by a Z-score transformation (subtract mean, divide by std dev.), min/max normalizer,
etc.

"""


# Directory where all project data files are to be stored
proj_dir = "/projects/FRNN/frnn_loader"

# 1/ Describe the dataset
predictor_tags = [
    "q95",
    "efsli",
    "ipspr15V",
    "efsbetan",
    "efswmhd",
    "dusbradial",
    "dssdenest",
    "pradcore",
    "pradedge",
    "bmspinj",
    "bmstinj",
    "ipsiptargt",
    "ipeecoil",
]
predictor_list = [signal_0d(tag) for tag in predictor_tags]

# Contains a list of shots that are non-disruptive
shotlist_clear = "d3d_clear_100.txt"
# Contains a list of shots that are disruptive
shotlist_disrupt = "d3d_disrupt_100.txt"


# Instantiate the filter we use to crimp the shot times
ip_filter = filter_ip_thresh(0.2)
signal_ip = signal_0d("ipspr15V")
my_backend = backend_hdf5(proj_dir)
my_fetcher = fetcher_d3d_v1()


num_shots = 5
shotdict = {}

i = 0
with open(join(proj_dir, "..", "shot_lists", shotlist_clear), "r") as fp:
    for line in fp.readlines():
        # Convert shotnr to int and ttd to float
        shotnr, ttd = [trf(val) for trf, val in zip([int, float], line.split())]

        # Run the Ip filter over the current shot
        tb, data = my_backend.load(signal_ip.info, shotnr)
        tmin, tmax = ip_filter(tb, data)
        shotdict.update(
            {
                shotnr: {
                    "tmin": tmin,
                    "tmax": tmax,
                    "is_disruptive": False,
                    "t_disrupt": -1.0,
                }
            }
        )

        i += 1
        if i >= num_shots:
            break

i = 0
with open(join(proj_dir, "..", "shot_lists", shotlist_disrupt), "r") as fp:
    for line in fp.readlines():
        # Convert shotnr to int and ttd to float
        shotnr, ttd = [trf(val) for trf, val in zip([int, float], line.split())]
        # ttd is given in seconds in the text files. Convert it to milliseconds
        ttd = ttd * 1e3
        shotdict.update(
            {
                shotnr: {
                    "tmin": tmin,
                    "tmax": ttd,
                    "is_disruptive": True,
                    "t_disrupt": ttd,
                }
            }
        )

        i += 1
        if i >= num_shots:
            break

print("shotdict = ", shotdict)


#########################################################################################################
#
# Next we create a list of datasets for all shots.
# The shots are cut to the time intervals defined by tmin and tmax
# No transformation has been defined,

dset_list = []
for shotnr in shotdict.keys():
    print(shotnr)

    # Resample all signals over the valid intervals
    my_resampler = resampler_causal(0.0, shotdict[shotnr]["tmax"], 1.0)

    ds = shot_dataset_disk(
        shotnr,
        predictors=predictor_list,
        resampler=my_resampler,
        backend_file=my_backend,
        fetcher=my_fetcher,
        root=proj_dir,
        download=True,
        dtype=torch.float32,
    )

    dset_list.append(ds)

#########################################################################################################
#
# With all datasets cropped to the correct time in place we continue by calculating the normalization.
# Do this using multi-processing
my_normalizer = mean_std_normalizer()
my_normalizer.fit(dset_list)

# Now we can add the normalizer as a transform. The __getitem__ method of the dataset
# will apply the transform.

# Verify that the returned data is about zero mean and order unity std deviation
for ds in dset_list:
    ds.transform = my_normalizer

    print(ds[:].shape, ds[:].mean(axis=0), ds[:].std(axis=0))


# end of file build_dataset.py
