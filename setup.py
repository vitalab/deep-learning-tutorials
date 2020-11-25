#!/usr/bin/env python
import builtins
import os
import pathlib

from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
PATH_ROOT = pathlib.Path(__file__).parent
builtins.__VITAL_SETUP__ = True


def load_requirements(
    path_dir=PATH_ROOT, file_name="requirements.txt", comment_char="#"
):  # noqa: D103
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def load_long_description():  # noqa: D103
    text = open(PATH_ROOT / "README.md", encoding="utf-8").read()
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace(".svg", ".png")
    return text


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="deep-learning-tutorials",
    version="0.0.1",
    description="Deep Learning Tutorials from the Videos & Images Theory and Analytics Laboratory at UdeS",
    author="Nathan Painchaud",
    author_email="nathan.painchaud@usherbrooke.ca",
    url="https://github.com/nathanpainchaud/deep-learning-tutorials",
    license="MIT",
    packages=find_packages(),
    long_description=load_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    setup_requires=[],
    install_requires=load_requirements(),
)
