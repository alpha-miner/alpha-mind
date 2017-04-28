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
    print("Parameters(n_samples: {0}, n_features: {1}, n_loops: {2})".format(n_samples, n_features, n_loops))

    y = np.random.randn(n_samples)
    x = np.random.randn(n_samples, n_features)

    start = dt.datetime.now()
    for _ in range(n_loops):
        _ = neutralize(x, y)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    for _ in range(n_loops):
        benchmark_model = LinearRegression(fit_intercept=False)
        benchmark_model.fit(x, y)
        _ = y - x @ benchmark_model.coef_
    benchmark_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))

if __name__ == '__main__':
    benchmark_neutralize(3000, 10, 1000)
