![pipeline-status](https://gitlab.com/gitlab-org/gitlab/badges/main/pipeline.svg)
![coverage](https://gitlab.com/gitlab-org/gitlab/badges/main/coverage.svg?job=coverage)

# Outdated
As of 2023-03, this code is no longer maintained. Please use [d3d_loaders](https://github.com/PlasmaControl/d3d_loaders).
Examples on how to use `d3d_loaders` are in


# frnn-loader

Code in this repository makes big fusion data available as [pytorch](https://pytorch.org/) style
[Datasets](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), targeting deep-learning
tasks.

[Documentation](https://rkube.git.pppl.gov/frnn-loader/index.html)

## Usage

To define a Dataset for one D3D discharge using filterscope, q95, and pedestal electron density signals, a dataset can be instantiated like this:

```python
import torch

from frnn_loader.primitives.resamplers import resampler_causal
from frnn_loader.backends.backend_txt import backend_txt
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.loaders.frnn_dataset import shot_dataset

my_resampler = resampler_causal(0.0, 2.0, 1e-3)

# Instantiate a file backend
my_backend_file = backend_txt("/home/rkube/datasets/frnn/signal_data_new_2021/")
my_fetcher = fetcher_d3d_v1()

signal_fs07 = signal_0d("fs07")
signal_q95 = signal_0d("q95")


ds = shot_dataset(184800, [signal_fs07, signal_q95], 
                    resampler=my_resampler, backend_file=my_backend_file, 
                    backend_fetcher=my_fetcher, download=True,
                    dtype=torch.float32)

for item in ds:
    pass
```

Here `frnn_loader` handles details of 
* resampling the signals onto a common time-base
* Caching the data from respective MDS servers into local file automatically in the backgroun.

## Installation
To install the `frnn_loaders` package in your conda environment do

```bash
git clone https://git.pppl.gov/rkube/frnn-loader
cd frnn_loader
pip install -e .
```


## Contributing
To contribute, please fork the repository and send pull requests. Only pull requests where all unit
tests pass will be considered.

The unit tests can be run as
```python
python -m unittest tests
```

Some unit tests will connect to MDS servers to download datasets. Please make sure to run them
on systems that are whitelisted, such as `traverse.princeton.edu` or on `portal.pppl.gov`.

## Coding Standard

A git pre-commit hook will run (Black)[https://github.com/psf/black] to automatically format all code.
It is encouraged to follow those guidelines for
* Syntax: [PEP-8](https://peps.python.org/pep-0008/)
* Comments: [google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)


## Support
For questions, please contact Ralph Kube: rkube@pppl.gov


## Authors and acknowledgment
This project is derived from the [FRNN project](https://github.com/PPPLDeepLearning/plasma-python)
Original authors: Julien Kates-Harbeck, Alexey Svyatkovskiy, Kyle Felker



## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.



Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.







## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.




