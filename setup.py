#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
import os

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name='othoz_adding_sum',
    author='narendhrancs@gmail.com',
    description='Othoz ML take home test',
    version='0.1',
    packages=['src'],
    package_data={
        '': ['*.py','*.txt'],
        'requirements.txt':['*.txt'],
        'cfg':['*.cfg']
    },
    scripts=[os.path.join('src', 'main.py')],
    install_requires=install_requires,
    python_requires='>=3.6'
)