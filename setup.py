# -*- coding: utf-8 -*-
"""
Created on 2017-4-25
@author: cheng.li
"""
import platform
import sys
from setuptools import setup
from setuptools import find_packages
from distutils.extension import Extension
import numpy as np
import Cython
from Cython.Build import cythonize
Cython.Compiler.Options.annotate = True

VERSION = "0.1.0"

if "--line_trace" in sys.argv:
    line_trace = True
    print("Build with line trace enabled ...")
    sys.argv.remove("--line_trace")
else:
    line_trace = False
ext_modules = ['alphamind/cyimpl.pyx']


def generate_extensions(ext_modules, line_trace=False):
    extensions = []
    if line_trace:
        print("define cython trace to True ...")
        define_macros = [('CYTHON_TRACE', 1), ('CYTHON_TRACE_NOGIL', 1)]
    else:
        define_macros = []
    if platform.system() != "Windows":
        extra_compile_args = ['-O3', '-std=c++11']
    else:
        extra_compile_args = ['/Ox']
    for pyxfile in ext_modules:
        ext = Extension(name='.'.join(pyxfile.split('/'))[:-4],
                        sources=[pyxfile],
                        define_macros=define_macros,
                        extra_compile_args=extra_compile_args)
        extensions.append(ext)
    return extensions

if platform.system() != "Windows":
    import multiprocessing
    n_cpu = multiprocessing.cpu_count()
else:
    n_cpu = 0

ext_modules_settings = cythonize(generate_extensions(ext_modules, line_trace),
                                 compiler_directives={'embedsignature': True, 'linetrace': line_trace},
                                 nthreads=n_cpu)
setup(
    name='Alpha-Mind',
    version=VERSION,
    packages=find_packages(),
    url='',
    license='',
    author='wegamekinglc',
    author_email='',
    ext_modules=ext_modules_settings,
    include_dirs=[np.get_include()],
    description=''
)