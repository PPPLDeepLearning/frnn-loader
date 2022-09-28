# -*- coding: utf-8 -*-

"""FRNN Dataset """

import logging
import tempfile
from os.path import join, isdir
from os import remove

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
        # Pre-calculate the array shape. That is, the sum of the channels over all predictors
        self.sum_all_channels = sum([pred.num_channels for pred in self.predictors])
        # Add iteration capabilities
        self._current_index = 0

        # Create a temporary file for the dataset.
        # If we want to download we need to have a fetcher bassed
        if self.download:
            assert self.fetcher is not None

        print(f"__init__: root = {self.root}")
        assert isdir(self.root)

        # Create a temporary file name for HDF5 storage.
        # Note that this is not the data file that contains the signal data for a given shot.
        self.tmp_fname = join(self.root, f"{next(tempfile._get_candidate_names())}.h5")
        with h5py.File(self.tmp_fname, "a") as fp:
            # In the data-loading stage data will be signal by signal. The transformed and
            # resampled signals are stored in HDF5
            h5_grp_trf = fp.create_group("transformed")
            h5_grp_trf.attrs["shotnr"] = self.shotnr

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
                        tb, _, signal_data, _, _, _ = self.fetcher.fetch(
                            signal.info, self.shotnr
                        )
                        self.backend_file.store(
                            signal.info, self.shotnr, tb, signal_data
                        )
                    else:
                        raise err

                logging.info(
                    f"Loaded signal {signal}: tb.shape = {tb.shape}, signal.shape = {signal_data.shape}"
                )

                # 2nd step: Re-sample
                tb_rs, signal_data_rs = resampler(tb, signal_data)
                logging.info(
                    f"Resampled signal {signal}: tb.shape = {tb_rs.shape}, signal.shape = {signal_data_rs.shape}"
                )
                # # 3rd step: Transform
                # if transform is not None:
                #     signal_data_rs = self.transform(signal_data_rs)
                # logging.info(
                #     f"Transformed signal {signal}: tb.shape = {tb_rs.shape}, signal.shape = {signal_data_rs.shape}"
                # )

                # 4th step: store processed data in HDF5
                grp = h5_grp_trf.create_group(signal.info["LocalPath"] + "_trf")
                dset = grp.create_dataset(
                    "signal_data", signal_data_rs.shape, dtype="f"
                )
                dset[:] = signal_data_rs[:]
                dset = grp.create_dataset("timebase", tb_rs.shape, dtype="f")
                dset[:] = tb_rs[:]

    def delete_data_file(self):
        """Deletes the temporary datafile.

        This removes all remnants of this object that the garbage collector would not pick up.
        """
        remove(self.tmp_fname)

    def __len__(self):
        return len(self.resampler)

    def __getitem__(self, idx):
        """Fetches a single sample.

        Note: Performance could be improved by implementing slicing directly here:
        https://discuss.pytorch.org/t/dataloader-sample-by-slices-from-dataset/113005/5
        """
        if isinstance(idx, torch.Tensor):
            print("idx is a tensor")
            # Sorted indices
            sort_idx = torch.argsort(idx)
            idx_sorted = idx[sort_idx]

            # Using the argsort trick to get the original indices
            sort_idx2 = torch.argsort(sort_idx)
            idx_orig = idx[sort_idx[sort_idx2]]

            assert all(idx_orig == idx)

            idx = idx.tolist()
            idx_sorted = idx_sorted.tolist()

            # Number of elements to fetch

            print(idx_sorted)
            num_ele = len(idx)
        elif isinstance(idx, list):
            # Assume that idx is a list if indices
            # This code is the same as when idx is a torch.Tensor.
            # Sorted indices
            idx = torch.tensor(idx)
            sort_idx = torch.argsort(idx)
            idx_sorted = idx[sort_idx]

            # Using the argsort trick to get the original indices
            sort_idx2 = torch.argsort(sort_idx)
            idx_orig = idx[sort_idx[sort_idx2]]

            assert all(idx_orig == idx)

            idx = idx.tolist()
            idx_sorted = idx_sorted.tolist()

            # Number of elements to fetch
            num_ele = len(idx)

        elif isinstance(idx, slice):
            # This is kind of a hack.
            # When we pass a slice object as the index we
            # - Pass the slice for indexing the dataset. Assume that the slice is compatible with
            #   h5py's slicing https://docs.h5py.org/en/stable/high/dataset.html?highlight=slice#fancy-indexing
            # - We still need to calculate num_ele. In general, a slice can not define a length.
            #   Instead, the length of a slice is defined only with the array the slice is used to address.
            #   Use information from the time-base resampler as a proxy of the data length to calcultae
            #   num_ele here
            tb_dummy = torch.arange(
                self.resampler.t_start,
                self.resampler.t_end,
                self.resampler.dt,
                dtype=self.dtype,
            )
            num_ele = tb_dummy[idx].shape[0]
            idx_sorted = idx

        else:
            idx_sorted = idx
            num_ele = 1

        # HDF5 requires sorted indices for access. Argsorting twice can reverse one argsort:
        """
        # Array with random indices
        idx_orig = np.array([1, 2, 9, 3, 4, 8, 5])
        idx = np.zeros_like(idx_orig)
        idx[:] = idx_orig[:]

        # sort_idx is the array that induces an order on idx
        sort_idx = np.argsort(idx)
        print(sort_idx, idx[sort_idx])

        # Applying argsort to sort_idx gives us an array that allows us to re-construct the original shuffling
        sort_idx2 = np.argsort(sort_idx)
        print(idx[sort_idx[sort_idx2]])

        idx[sort_idx[sort_idx2]] == idx_orig
        """

        output = torch.zeros((num_ele, self.sum_all_channels), dtype=self.dtype)

        current_ch = 0
        with h5py.File(self.tmp_fname, "r") as fp:
            for pred in self.predictors:
                tb = fp[f"/transformed/{pred.info['LocalPath']}_trf"]["timebase"][
                    idx_sorted
                ]
                data = fp[f"/transformed/{pred.info['LocalPath']}_trf"]["signal_data"][
                    idx_sorted, :
                ]

                # Access pattern for 0d signals
                if isinstance(idx, list):
                    output[
                        :, current_ch : current_ch + pred.num_channels
                    ] = torch.tensor(data[sort_idx2.tolist(), :])
                elif isinstance(idx, slice):
                    output[
                        :, current_ch : current_ch + pred.num_channels
                    ] = torch.tensor(data)
                else:
                    output[0, current_ch : current_ch + pred.num_channels] = float(data)

                current_ch += pred.num_channels

        if self.transform:
            output = self.transform(output)

        return output

    def __iter__(self):
        """Iterator"""
        return self

    def __next__(self):
        """Implement iterable capability"""
        if self._current_index < self.__len__():
            rval = self.__getitem__(self._current_index)
            self._current_index += 1
            return rval

        raise StopIteration


# end of file frnn_dataset_disk.py
