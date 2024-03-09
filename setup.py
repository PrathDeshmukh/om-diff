#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="OM-Diff: A python package for inverse-design "
                "of organometallic catalysts with guided equivariant diffusion.",
    author="François Cornet",
    author_email="frjc@dtu.dk",
    url="https://github.com/frcnt/om-diff",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
        ]
    },
)
