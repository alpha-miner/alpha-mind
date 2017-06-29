# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import platform
import io
from setuptools import setup
from setuptools import find_packages
import numpy as np

VERSION = "0.1.0"

if platform.system() != "Windows":
    import multiprocessing
    n_cpu = multiprocessing.cpu_count()
else:
    n_cpu = 0

setup(
    name='Alpha-Mind',
    version=VERSION,
    packages=find_packages(),
    url='',
    license='MIT',
    author='wegamekinglc',
    author_email='',
    scripts=['alphamind/bin/alphamind'],
    install_requires=io.open('requirements.txt', encoding='utf8').read(),
    include_dirs=[np.get_include()],
    description=''
)