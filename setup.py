#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='othoz_adding_sum',
    author='narendhrancs@gmail.com',
    description='Othoz ML take home test',
    version='0.1.1',
    packages=find_packages(),
    scripts=[],
    install_requires=['tensorboard==2.11.0',
                      'torch==1.13.0',
                      'numpy==1.21.5',
                      'pandas==1.3.5',
                      'click',
                      'matplotlib'
                      ],
    python_requires='>=3.6',
    url='https://github.com/narendhrancs/othoz_adding_sum',
)