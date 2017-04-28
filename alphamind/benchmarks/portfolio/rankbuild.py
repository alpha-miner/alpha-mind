# -*- coding: utf-8 -*-
"""
Created on 2017-4-27

@author: cheng.li
"""

import datetime as dt
import numpy as np
import pandas as pd
from alphamind.portfolio.rankbuilder import rank_build


def benchmark_build_rank(n_samples: int, n_loops: int, n_included: int) -> None:
    print("-" * 60)
    print("Starting portfolio construction by rank benchmarking")
    print("Parameters(n_samples: {0}, n_included: {1}, n_loops: {2})".format(n_samples, n_included, n_loops))

    x = np.random.randn(n_samples, 1)

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = rank_build(x, n_included)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    for _ in range(n_loops):
        expected_weights = np.zeros((len(x), 1))
        expected_weights[(-x).argsort(axis=0).argsort(axis=0) < n_included] = 1. / n_included
    benchmark_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


def benchmark_build_rank_with_group(n_samples: int, n_loops: int, n_included: int, n_groups: int) -> None:
    print("-" * 60)
    print("Starting  portfolio construction by rank with group-by values benchmarking")
    print("Parameters(n_samples: {0}, n_included: {1}, n_loops: {2}, n_groups: {3})".format(n_samples, n_included, n_loops, n_groups))

    x = np.random.randn(n_samples, 1)
    groups = np.random.randint(n_groups, size=n_samples)

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = rank_build(x, n_included, groups=groups)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    for _ in range(n_loops):
        grouped_ordering = pd.DataFrame(-x).groupby(groups).rank()
        expected_weights = np.zeros((len(x), 1))
        masks = grouped_ordering <= n_included
        expected_weights[masks] = 1. / np.sum(masks)
    benchmark_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


if __name__ == '__main__':
    benchmark_build_rank(3000, 1000, 300)
    benchmark_build_rank_with_group(3000, 1000, 10, 30)
