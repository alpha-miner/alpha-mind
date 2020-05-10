# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

from alphamind.benchmarks.data.neutralize import benchmark_neutralize
from alphamind.benchmarks.data.neutralize import benchmark_neutralize_with_groups
from alphamind.benchmarks.data.standardize import benchmark_standardize
from alphamind.benchmarks.data.standardize import benchmark_standardize_with_group
from alphamind.benchmarks.data.winsorize import benchmark_winsorize_normal
from alphamind.benchmarks.data.winsorize import benchmark_winsorize_normal_with_group
from alphamind.benchmarks.portfolio.linearbuild import benchmark_build_linear
from alphamind.benchmarks.portfolio.percentbuild import benchmark_build_percent
from alphamind.benchmarks.portfolio.percentbuild import benchmark_build_percent_with_group
from alphamind.benchmarks.portfolio.rankbuild import benchmark_build_rank
from alphamind.benchmarks.portfolio.rankbuild import benchmark_build_rank_with_group
from alphamind.benchmarks.settlement.simplesettle import benchmark_simple_settle
from alphamind.benchmarks.settlement.simplesettle import benchmark_simple_settle_with_group

if __name__ == '__main__':
    benchmark_neutralize(3000, 10, 1000)
    benchmark_neutralize_with_groups(3000, 10, 1000, 30)
    benchmark_neutralize(30, 3, 50000)
    benchmark_neutralize_with_groups(30, 3, 50000, 3)
    benchmark_neutralize(50000, 50, 20)
    benchmark_neutralize_with_groups(50000, 50, 20, 50)
    benchmark_standardize(3000, 10, 1000)
    benchmark_standardize_with_group(3000, 10, 1000, 30)
    benchmark_standardize(100, 10, 50000)
    benchmark_standardize_with_group(100, 10, 5000, 4)
    benchmark_standardize(50000, 50, 20)
    benchmark_standardize_with_group(50000, 50, 20, 50)
    benchmark_winsorize_normal(3000, 10, 1000)
    benchmark_winsorize_normal_with_group(3000, 10, 1000, 30)
    benchmark_winsorize_normal(30, 10, 50000)
    benchmark_winsorize_normal_with_group(30, 10, 5000, 5)
    benchmark_winsorize_normal(50000, 50, 20)
    benchmark_winsorize_normal_with_group(50000, 50, 20, 50)
    benchmark_build_rank(3000, 1000, 300)
    benchmark_build_rank_with_group(3000, 1000, 10, 30)
    benchmark_build_rank(30, 50000, 3)
    benchmark_build_rank_with_group(30, 50000, 1, 3)
    benchmark_build_rank(50000, 20, 3000)
    benchmark_build_rank_with_group(50000, 20, 10, 300)
    benchmark_build_percent(3000, 1000, 0.1)
    benchmark_build_percent_with_group(3000, 1000, 0.1, 30)
    benchmark_build_percent(30, 50000, 0.1)
    benchmark_build_percent_with_group(30, 50000, 0.1, 3)
    benchmark_build_percent(50000, 20, 0.1)
    benchmark_build_percent_with_group(50000, 20, 0.1, 300)
    benchmark_build_linear(100, 3, 100)
    benchmark_build_linear(1000, 30, 10)
    benchmark_simple_settle(3000, 10, 1000)
    benchmark_simple_settle_with_group(3000, 10, 1000, 30)
    benchmark_simple_settle(30, 10, 50000)
    benchmark_simple_settle_with_group(30, 10, 50000, 5)
    benchmark_simple_settle(50000, 50, 20)
    benchmark_simple_settle_with_group(50000, 50, 20, 50)
