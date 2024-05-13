#!/usr/bin/env python

import sys
import os
import glob

from distutils.core import setup
from distutils.command.install_data import install_data
from setuptools.command.install import install

# read the long description
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'README.md'),encoding='utf-8') as f:
    long_description = f.read()

# read the package requirements
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),'requirements.txt'),encoding='utf-8') as f:
    install_requires = f.read().splitlines()

setup(name='GridNext',
      version='0.0.1',
      description='PyTorch based deep learning datasets and models for Spatial Transcriptomics (ST) data',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author="Aidan Daly",
      author_email="adaly@nygenome.org",
      url='https://github.com/adaly/gridnext',
      license="BSD",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD 3-Clause "New" or "Revised" License (BSD-3-Clause)',
          'Programming Language :: Python :: 3'],
      packages=['gridnext'],
      install_requires=install_requires,
      cmdclass={
          'install': install
      },
      include_package_data=True
)