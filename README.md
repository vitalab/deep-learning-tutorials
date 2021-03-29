# Deep Learning Tutorials
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Setup

### Virtual Environment
If you don't operate inside a virtual environment, or only have access to an incompatible python version (<3.8), it is
recommended you create a virtual environment using [`conda`](https://docs.conda.io/en/latest/):
```shell script
conda env create -f environment.yml
conda activate deep-learning-tutorials
```
Creating the environment this way also takes care of installing the dependencies for you, so you can skip the rest of
the setup and dive straight into one of the tutorials.

### Installing Dependencies
If you already have a python environment set aside for this project and just want to install the dependencies, you can
do that using the following command:
```shell script
pip install -e .
```


## How to Run
Once you've went through the [setup](#setup) instructions above, you can start exploring the tutorial's notebooks.
We recommend using JupyterLab to run the notebooks, which can be launched by running (from within your environment):
```shell script
jupyter-lab
```
When you've launched JupyterLab's web interface, you can simply navigate to any of the
[tutorials listed below](#available-tutorials), and follow the instructions in there!


## Available Tutorials

### Representation Learning
- [Basic Variational Autoencoders](tutorials/mnist-autoencoders.ipynb)
- [Variational Autoencoders Applied to Cardiac MRI](tutorials/cardiac-mri-autoencoders.ipynb)

You may download the MNIST and ACDC datasets [here](http://info.usherbrooke.ca/pmjodoin/projects/data.tar.gz).  Once downloaded, you may untar the file and copy the **data/** folder in the root of your code, at the same level than the **src/** and the **tutorials/** folders.

## How to Contribute
If you want to contribute to the project, then you have to install development dependencies and pre-commit hooks, on
top of the basic setup for using the project, detailed [above](#setup). The pre-commit hooks are there to ensure that
any code committed to the repository meets the project's format and quality standards.
```shell script
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```
