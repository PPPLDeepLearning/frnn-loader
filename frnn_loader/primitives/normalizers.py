# -*- coding: utf-8 -*-


# Calculate E[X] for every dataset
# Something is fishy with torch. So roll with numpy for now

import torch
import numpy as np
import multiprocessing as mp


class mean_std_normalizer:
    def __init__(self):
        self.mean = None
        self.var = None

    def fit(self, ds_list):
        """Fits Z-scores

        Fits E[X] and E[(X - E[X])^2] in a multi-pass way.
        The implementation makes use of threads using the multiprocessing package.

        Args
            ds_list - List of datasets
        """

        def calc_mean_single(ds_data, done_q):
            # Calculates the mean channel-wise for a single shot
            result = np.array(ds_data.mean(axis=0))
            done_q.put(result)

        def calc_var_single(ds_data, mean_all, done_q):
            # Calculate E[(X - E[X])^2] channel-wise for a single shot
            result = np.array(((ds_data - mean_all) ** 2.0).mean(axis=0))
            done_q.put(result)

        # Instantiate a queue. A group of worker threads executes calc_mean_single
        # on a single dataset and writes the result into the Queue.
        Q = mp.Queue()
        processes = []
        for ds in ds_list:
            p = mp.Process(target=calc_mean_single, args=(ds[:].numpy(), Q))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Once all threads are finished, the master thread pulls the results from the Queue
        mean_list = []
        while True:
            try:
                item = Q.get_nowait()
            except:
                break
            mean_list.append(item)

        # The mean over all threads is the just the average over E[X] of the individual shots.
        mean_all = np.stack(mean_list, axis=0).mean(axis=0)

        # Proceed by calculating E[(X - E[X])^2] for every dataset.
        # The approach is the same as for calc_mean_single.
        processes = []
        for ds in ds_list:
            p = mp.Process(target=calc_var_single, args=(ds[:].numpy(), mean_all, Q))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        var_list = []
        while True:
            try:
                item = Q.get_nowait()
            except:
                break
            var_list.append(item)

        # This is the standard deviation, sqrt(E[(X - E[X])^2])
        var_all = np.stack(var_list, axis=0).mean(axis=0)

        # Save results as torch.tensor
        self.mean_all = torch.tensor(mean_all)
        self.std_all = torch.tensor(np.sqrt(var_all))

    def __call__(self, X):
        """Returns normalized values.

        Args
            X (array-like) Un-normalized data

        Returns:
            X_norm (array_like) - Normalized data
        """
        return (X - self.mean_all) / self.std_all


# end of file normalizers.py
