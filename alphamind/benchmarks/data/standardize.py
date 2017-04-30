# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from alphamind.data.standardize import standardize


def benchmark_standardize(n_samples: int, n_features: int, n_loops: int) -> None:
    print("-" * 60)
    print("Starting standardizing benchmarking")
    print("Parameters(n_samples: {0}, n_features: {1}, n_loops: {2})".format(n_samples, n_features, n_loops))

    x = np.random.randn(n_samples, n_features)

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = standardize(x)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = zscore(x)
    benchmark_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


def benchmark_standardize_with_group(n_samples: int, n_features: int, n_loops: int, n_groups: int) -> None:
    print("-" * 60)
    print("Starting standardizing with group-by values benchmarking")
    print("Parameters(n_samples: {0}, n_features: {1}, n_loops: {2}, n_groups: {3})".format(n_samples, n_features, n_loops, n_groups))

    x = np.random.randn(n_samples, n_features)
    groups = np.random.randint(n_groups, size=n_samples)

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = standardize(x, groups=groups)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = pd.DataFrame(x).groupby(groups).transform(lambda s: (s - s.mean(axis=0)) / s.std(axis=0))
    benchmark_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


if __name__ == '__main__':
    benchmark_standardize(3000, 10, 1000)
    benchmark_standardize_with_group(3000, 10, 1000, 30)
