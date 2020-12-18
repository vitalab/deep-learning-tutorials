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
the setup and dive straight into one of the tutorials!

### Installing Dependencies
If you already have a python environment set aside for this project and just want to install the dependencies, you can
do that using the following command:
```shell script
pip install -e .
```


### Installing Jupyter Widgets
To get some interactive features, e.g. progress bars, to display properly in JupyterLab, you also need to install
and/or activate the required widgets, by following the instructions below:
```shell script
# Setup ipywidgets to work with JupyterLab
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```


## How to Run
Once you've went through the [setup](#setup) instructions above, you can start exploring the notebook the tutorials.
We recommend using JupyterLab to run the notebooks, which can be launched by running (from within your environment):
```shell script
jupyter-lab
```
When you've launched JupyterLab's web interface, you can simply navigate to any of the
[tutorials listed below](#available-tutorials), and follow the instructions in there!


## Available Tutorials


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
