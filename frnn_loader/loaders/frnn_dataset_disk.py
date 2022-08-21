# -*- coding: utf-8 -*-

"""FRNN Dataset """

import logging
import tempfile
from os.path import join

import h5py
import torch
from torch.utils.data import Dataset

from frnn_loader.primitives.signal import signal_0d
from frnn_loader.utils.errors import SignalCorruptedError, NotDownloadedError


class shot_dataset_disk(Dataset):
    """Dataset representing a shot which uses disk data as storage

    This dataset maps the __getitem__ method to a pre-cached version on disk.
    The cached version on disk is built
    * Using the predictors and targets
    * With all transformations applied to them.

    Args
    * shotnr (int)
    * predictors (list(str))
    * resampler ~frnn_loader.primitives.resampler
    * backend_file ~frnn_loader.backends.backend
    * fetcher (~frnn_loader.backends.fetcher)
    * cache
    * root (string) - Directory where transformed data is stored
    * download (bool) - If true, downloadin missing data
    * transform (transform) - Transformations applied to data
    * dtype

    """

    def __init__(
        self,
        shotnr,
        predictors,
        resampler,
        backend_file,
        fetcher,
        root,
        download=False,
        transform=None,
        dtype=torch.float32,
    ):
        """Initializes the disk dataset."""
        self.shotnr = shotnr
        self.predictors = predictors
        self.resampler = resampler
        self.backend_file = backend_file
        self.fetcher = fetcher
        self.root = root
        self.download = download
        self.transform = transform
        self.dtype = dtype

        # Create a temporary file for the dataset.

        # If we want to download we need to have a fetcher bassed
        if self.download:
            assert self.fetcher is not None

        # Create a temporary file name for HDF5 storage
        self.tmp_fname = join(self.root, f"{next(tempfile._get_candidate_names())}.h5")
        with h5py.File(self.tmp_fname, "a") as fp:
            # In the data-loading stage data will be signal by signal. The transformed and
            # resampled signals are stored in HDF5
            h5_grp_trf = fp.create_group("transformed")

            # Next is pre-processing. We attack it like this
            # 1. Fetch the signals
            # 2. Re-sampled the signals
            # 3. Apply the transformation
            # 4. Store the data in a hdf5 file.

            invalid_signals = 0  # Count number of invalid signals
            for signal in self.predictors:
                # 1st step: Fetch the data
                try:
                    tb, signal_data = self.backend_file.load(signal.info, self.shotnr)

                except SignalCorruptedError as err:
                    logging.error(f"SignalCorrupted occured: {err}")
                    invalid_signals += 1
                    raise (err)

                except NotDownloadedError as err:
                    logging.error(f"Signal not downloaded: {err}")
                    if self.download:
                        logging.info(f"Downloading signal {signal}")
                        tb, _, signal_data, _, _, _ = self.backend_fetcher.fetch(
                            signal.info, self.shotnr
                        )
                        self.backend_file.store(
                            signal.info, self.shotnr, tb, signal_data
                        )
                    else:
                        raise err

                logging.info(
                    f"Loaded signal {signal}: tb.shape = "
                    + str(tb.shape)
                    + ", signal.shape = "
                    + str(signal_data.shape)
                )

                # 2nd step: Re-sample
                tb_rs, signal_data_rs = resampler(tb, signal_data)
                # 3rd step: Transform
                if transform is not None:
                    signal_data = self.transform(signal_data)

                # 3rd step: store processed data in HDF5
                grp = h5_grp_trf.create_group(signal.info["LocalPath"])
                dset = grp.create_dataset(
                    "signal_data", signal_data_rs.shape, dtype="f"
                )
                dset[:] = signal_data_rs[:]
                dset = grp.create_dataset("timebase", tb_rs.shape, dtype="f")
                dset[:] = tb_rs[:]


# end of file frnn_dataset_disk.py
