# -*- coding: utf-8 -*-
#
import logging
from math import ceil
import numpy as np


class random_sequence_sampler:
    """Samples random sequences from a list of shot datasets.

    A list of shots defines the data in this shot.
    To iterate over this shots, we draw sequences of fixed length,
    but random start from these shots.

    The probability that a single sequence is from a given shot is
    given by length(shot) / sum(length(all_shot))
    Here length(shot) is just the number of time samples for this shot.

    Typically, shots have different length. We define a complete
    iteration over the dataset as the number of samples of length `seq_length`
    divided by the length of the longest shot.

    Args:
        ds_list (frnn_multi_dataset) : Multi-shot dataset to load data from
        seq_length (int) : Length of sequences
        batch_size (int, default=1) : Batch size
        drop_last (bool, default=True) : If the set to True to drop the last
            incomplete batch, if the dataset size is not divisible by the batch
            size. If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
    """

    def __init__(self, ds_list, seq_length, batch_size=1, drop_last=True):
        # This is the length of all shots
        self.shot_length = [len(ds) for ds in ds_list]
        self.num_shots = len(ds_list)
        self.seq_length = seq_length
        # self.batch_size = batch_size
        # self.drop_last = drop_last

        # The probability of drawing a sequence from a given shot is
        # given by the shots length relative to the cumulative length of
        # all shots
        self.p_shot = [len(ds) / sum(self.shot_length) for ds in ds_list]

        # Number of draws to exhaust the iterator
        self.num_draws = sum(self.shot_length) // self.seq_length

    def __iter__(self):
        """Returns a tuple of a single shot index and slice

        The iterator returns a total of
            sum(self.shot_length) // self.seq_length
        batches

        Each sample has dimension
            (self.batch_size, self.seq_length, shot.sum_all_channels)

        The probability that a sample is from shot s is given by
            length(shot[s]) / sum(length(shot[]))
        """
        # Determine the shots that we get slices from
        shot_idx = np.random.choice(
            np.arange(self.num_shots), (self.num_draws,), p=self.p_shot
        )
        # Upper index for the sequence to start, adapted to the shot we draw from
        upper_idx = [self.shot_length[s] - self.seq_length for s in shot_idx]
        # Upper index for the slice
        start_idx = [np.random.randint(0, u) for u in upper_idx]

        # Return a single index
        for i in range(self.num_draws):
            yield (
                shot_idx[i],
                slice(start_idx[i], start_idx[i] + self.seq_length, 1),
            )

    def __len__(self) -> int:
        return max(self.shot_length) // self.seq_length


class batched_random_sequence_sampler(random_sequence_sampler):
    """Random sequence sampler, but with mini-batches"""

    def __init__(self, ds_list, seq_length, batch_size=1, drop_last=True):
        super(batched_random_sequence_sampler, self).__init__(ds_list, seq_length)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        """Returns a list of tuples of shot indices and slices

        The iterator returns a total of
            sum(self.shot_length) // self.seq_length
        batches

        Each sample has dimension
            (self.batch_size, self.seq_length, shot.sum_all_channels)

        The probability that a sample is from shot s is given by
            length(shot[s]) / sum(length(shot[]))
        """
        # Determine the shots that we get slices from
        shot_idx = np.random.choice(
            np.arange(self.num_shots), (self.num_draws,), p=self.p_shot
        )
        # Upper index for the sequence to start, adapted to the shot we draw from
        upper_idx = [self.shot_length[s] - self.seq_length for s in shot_idx]

        # Upper index for the slice
        start_idx = [np.random.randint(0, u) for u in upper_idx]

        # When using batch_size > 2, draw multiple samples, put them
        # in a list and let collate_fn deal with the rest
        for i in range(0, self.num_draws, self.batch_size):
            yield [
                [
                    shot_idx[i + b],
                    slice(start_idx[i + b], start_idx[i + b] + self.seq_length, 1),
                ]
                for b in range(self.batch_size)
                # Filter to drop some elements when approaching upper range of iteration
                if (i + b) < self.num_draws
            ]

        # Handle final samples if drop_last=False
        if self.drop_last == False and (
            ceil(self.num_draws / self.batch_size) > (self.num_draws // self.batch_size)
        ):
            start = self.num_draws // self.batch_size * self.batch_size
            stop = self.num_draws
            yield [
                (
                    shot_idx[i],
                    slice(start_idx[i], start_idx[i] + self.seq_length, 1),
                )
                for i in range(start, stop)
            ]


class frnn_loader:
    """Custom loader for FRNN datasets.

    PyTorch expects training batches to have shape
    (nbatch, seq_length, ndim)
    with
    * nbatch - The batch size
    * seqlength - Length of time-series sequences
    * ndim - Sum of all diagnostic channels.

    This loader performs random sampling for a shot_dataset_disk

    """

    def __init__(self, dataset, batch_size, seq_length, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.collate_fn = collate_fn

    def __iter__(self):
        """Reset iteration index and return self"""
        self.index = 0
        return self

    def __next__(self):
        """Get next item in iteration"""
        if self.index >= len(self.dataset):
            raise StopIteration

        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        return self.collate_fn([self.get() for _ in range(batch_size)])

    def get(self):
        item = self.dataset[self.index]
        self.index += 1
        return item
