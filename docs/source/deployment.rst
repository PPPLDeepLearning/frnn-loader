
Deployment
==========


traverse
--------

To deploy `frnn-loader` on traverse we first need to create a conda environment and activate it.

    $ conda create --name frnn python=3.10
    # conda activate frnn

Pytorch can be installed from `source https://github.com/pytorch/pytorch#from-source`_ :

    $ git clone --recursive https://github.com/pytorch/pytorch
    $ pip install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses

Compilation works fine with cuda-11.3 (as of 2022-08):

    $ module list
    Currently Loaded Modulefiles:
    1) cudatoolkit/11.3   2) cudnn/cuda-11.x/8.2.0   3) anaconda3/2021.11 


But the build script doesn't pick up on cuDNN, some paths have to be set manually

    $ export CUDNN_LIB_DIR=/usr/local/cudnn/cuda-11.3/8.2.0/lib64
    $ export CUDNN_INCLUDE_DIR=/usr/local/cudnn/cuda-11.3/8.2.0/include

Then compile pytorch:

    $ cd pytorch
    $ python setup.py install

Using pip, scipy can't be installed because there is a problem with openblas. Install scipy using conda:

    $ conda install scipy

Finally, fetch `frnn-loader`, install the requirements and the package

    $ cd ..
    $ git clone git@git.pppl.gov:rkube/frnn-loader.git
    $ cd frnn-loader
    $ pip install -r requirements.txt 
    $ python setup.py -e install





