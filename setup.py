# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import platform
import io
import os
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
                  include_dirs=["alphamind/pfopt/include/clp",
                                "alphamind/pfopt/include/ipopt",
                                "alphamind/pfopt/include/pfopt",
                                "alphamind/pfopt/include/eigen",
                                "alphamind/pfopt/include/alglib"],
                  libraries=['pfopt'],
                  library_dirs=['alphamind/pfopt/lib'],
                  extra_compile_args=['-std=c++11']),
        ]

    lib_files = []
    lib_folder = 'alphamind/pfopt/lib'
    for file in os.listdir(lib_folder):
        lib_files.append(os.path.join(lib_folder, file))

else:
    extensions = [
        Extension('alphamind.cython.optimizers',
                  ['alphamind/cython/optimizers.pyx'],
                  include_dirs=["alphamind/pfopt/include/clp",
                                "alphamind/pfopt/include/ipopt",
                                "alphamind/pfopt/include/pfopt",
                                "alphamind/pfopt/include/eigen",
                                "alphamind/pfopt/include/alglib"],
                  libraries=['pfopt', 'alglib', 'libClp', 'libCoinUtils', 'libipopt', 'libcoinhsl', 'libcoinblas', 'libcoinlapack', 'libcoinmetis'],
                  library_dirs=['alphamind/pfopt/lib'],
                  extra_compile_args=['/MD']),
    ]

    lib_files = []


lib_files = []
lib_folder = 'alphamind/pfopt/lib'
for file in os.listdir(lib_folder):
    lib_files.append(os.path.join(lib_folder, file))

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
    data_files=[('alphamind/pfopt/lib', lib_files)],
    description='',
    include_package_data=True
)