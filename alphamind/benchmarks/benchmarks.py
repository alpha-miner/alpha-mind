# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

from alphamind.benchmarks.neutralize import benchmark_neutralize
from alphamind.benchmarks.standardize import benchmark_standardize
from alphamind.benchmarks.standardize import benchmark_standardize_with_group
from alphamind.benchmarks.winsorize import benchmark_winsorize_normal
from alphamind.benchmarks.winsorize import benchmark_winsorize_normal_with_group


if __name__ == '__main__':

    benchmark_neutralize(3000, 10, 1000)
    benchmark_neutralize(30, 10, 50000)
    benchmark_standardize(3000, 10, 1000)
    benchmark_standardize_with_group(3000, 10, 100, 30)
    benchmark_standardize(30, 10, 50000)
    benchmark_standardize_with_group(30, 10, 5000, 5)
    benchmark_winsorize_normal(30, 10, 50000)
    benchmark_winsorize_normal_with_group(30, 10, 5000, 5)
    benchmark_winsorize_normal(30, 10, 50000)
    benchmark_winsorize_normal_with_group(30, 10, 5000, 5)
