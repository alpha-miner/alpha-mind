# -*- coding: utf-8 -*-
"""
Created on 2017-4-25

@author: cheng.li
"""

import datetime as dt

import numpy as np
from sklearn.linear_model import LinearRegression

from alphamind.data.neutralize import neutralize


def benchmark_neutralize(n_samples: int, n_features: int, n_loops: int) -> None:
    print("-" * 60)
    print("Starting least square fitting benchmarking")
    print("Parameters(n_samples: {0}, n_features: {1}, n_loops: {2})".format(n_samples, n_features,
                                                                             n_loops))

    y = np.random.randn(n_samples, 5)
    x = np.random.randn(n_samples, n_features)

    start = dt.datetime.now()
    for _ in range(n_loops):
        calc_res = neutralize(x, y)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    for _ in range(n_loops):
        benchmark_model = LinearRegression(fit_intercept=False)
        benchmark_model.fit(x, y)
        exp_res = y - x @ benchmark_model.coef_.T
    benchmark_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))

    np.testing.assert_array_almost_equal(calc_res, exp_res)


def benchmark_neutralize_with_groups(n_samples: int, n_features: int, n_loops: int,
                                     n_groups: int) -> None:
    print("-" * 60)
    print("Starting least square fitting with group benchmarking")
    print(
        "Parameters(n_samples: {0}, n_features: {1}, n_loops: {2}, n_groups: {3})".format(n_samples,
                                                                                          n_features,
                                                                                          n_loops,
                                                                                          n_groups))
    y = np.random.randn(n_samples, 5)
    x = np.random.randn(n_samples, n_features)
    groups = np.random.randint(n_groups, size=n_samples)

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = neutralize(x, y, groups)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()

    model = LinearRegression(fit_intercept=False)
    for _ in range(n_loops):
        for i in range(n_groups):
            curr_x = x[groups == i]
            curr_y = y[groups == i]
            model.fit(curr_x, curr_y)
            _ = curr_y - curr_x @ model.coef_.T
    benchmark_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


if __name__ == '__main__':
    benchmark_neutralize(3000, 10, 1000)
    benchmark_neutralize_with_groups(3000, 10, 1000, 30)
