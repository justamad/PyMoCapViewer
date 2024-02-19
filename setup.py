#!/usr/bin/env python

from distutils.core import setup

import setuptools

setup(
    name='PyMoCapViewer',
    version='0.0.14',
    description='A not anymore so simple MoCap Viewer',
    author='Justin Albert',
    author_email='justin.albert@hpi.de',
    url='https://github.com/justamad/PyMoCapViewer',
    packages=setuptools.find_packages(),
    install_requires=['vtk', 'open3d', 'numpy', 'pandas', 'scipy'],
)
