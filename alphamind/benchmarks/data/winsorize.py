# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import datetime as dt
import numpy as np
import pandas as pd
from alphamind.data.winsorize import winsorize_normal


def benchmark_winsorize_normal(n_samples: int, n_features: int, n_loops: int) -> None:
    print("-" * 60)
    print("Starting winsorize normal benchmarking")
    print("Parameters(n_samples: {0}, n_features: {1}, n_loops: {2})".format(n_samples, n_features, n_loops))

    num_stds = 2

    x = np.random.randn(n_samples, n_features)

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = winsorize_normal(x, num_stds)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    def impl(x):
        std_values = x.std(axis=0)
        mean_value = x.mean(axis=0)

        lower_bound = mean_value - num_stds * std_values
        upper_bound = mean_value + num_stds * std_values

        res = np.where(x > upper_bound, upper_bound, x)
        res = np.where(res < lower_bound, lower_bound, res)
        return res

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = impl(x)
    benchmark_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


def benchmark_winsorize_normal_with_group(n_samples: int, n_features: int, n_loops: int, n_groups: int) -> None:
    print("-" * 60)
    print("Starting winsorize normal with group-by values benchmarking")
    print("Parameters(n_samples: {0}, n_features: {1}, n_loops: {2}, n_groups: {3})".format(n_samples, n_features, n_loops, n_groups))

    num_stds = 2

    x = np.random.randn(n_samples, n_features)
    groups = np.random.randint(n_groups, size=n_samples)

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = winsorize_normal(x, num_stds, groups=groups)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    def impl(x):
        std_values = x.std(axis=0)
        mean_value = x.mean(axis=0)

        lower_bound = mean_value - num_stds * std_values
        upper_bound = mean_value + num_stds * std_values

        res = np.where(x > upper_bound, upper_bound, x)
        res = np.where(res < lower_bound, lower_bound, res)
        return res

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = pd.DataFrame(x).groupby(groups).transform(impl)
    benchmark_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


if __name__ == '__main__':
    benchmark_winsorize_normal(3000, 10, 1000)
    benchmark_winsorize_normal_with_group(3000, 10, 1000, 30)
