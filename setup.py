# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import io
from setuptools import setup
from setuptools import find_packages

VERSION = "0.3.1"

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
    description='',
    include_package_data=True
)