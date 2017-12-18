# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import platform
import io
from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

VERSION = "0.1.0"

if platform.system() != "Windows":
    import multiprocessing
    n_cpu = multiprocessing.cpu_count()
else:
    n_cpu = 0


if platform.system() != "Windows":
    extensions = [
        Extension('alphamind.cython.optimizers',
                  ['alphamind/cython/optimizers.pyx'],
                  include_dirs=["./libs/include/linux/clp",
                                "./libs/include/linux/ipopt",
                                "./libs/include/linux/pfopt",
                                "./libs/include/linux/eigen",
                                "./libs/include/linux/alglib"],
                  libraries=['pfopt'],
                  library_dirs=['./libs/lib/linux'],
                  extra_compile_args=['-std=c++11']),
        ]
else:
    extensions = [
        Extension('alphamind.cython.optimizers',
                  ['alphamind/cython/optimizers.pyx'],
                  include_dirs=["./libs/include/windows/clp",
                                "./libs/include/windows/ipopt",
                                "./libs/include/windows/pfopt",
                                "./libs/include/windows/eigen",
                                "./libs/include/windows/alglib"],
                  libraries=['pfopt', 'alglib', 'libClp', 'libCoinUtils', 'libipopt', 'libcoinhsl', 'libcoinblas', 'libcoinlapack', 'libcoinmetis'],
                  library_dirs=['./libs/lib/windows'],
                  extra_compile_args=['/MD']),
    ]

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
    ext_modules=cythonize(extensions),
    description=''
)