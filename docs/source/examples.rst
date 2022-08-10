
Example
=======

Create a Dataset consisting of signal `fs07` for D3D discharge 184800.

.. code-block:: 
    
        # Re-sample the signal onto the interval [0.0:2.0] seconds, using a time-step of 1ms:
        my_resampler = resampler_last(0.0, 2.0, 1e-3)

        # Instantiate a file backend. This implies that all signals are stored in the 
        # path pointed to by root:
        root = "/home/rkube/datasets/frnn/signal_data_new_2021/"
        my_backend_file = backend_txt(root)

        # Define the signal "fs07":
        signal_fs07 = signal_0d("fs07")

        # Instantiate a dataset
        ds = shot_dataset(184800, [signal_fs07], resampler=my_resampler, backend_file=my_backend_file, 
                          download=False, dtype=torch.float32)

        # We can now iterate over the datasets samples:
        for item in ds:
            # Do things

This example assumes that all signal data is available in "/home/rkube/datasets/frnn/signal_data_new_2021".
Data loading from disk is handled using a :class:`frnn_loader.backends.backend_txt` object.


