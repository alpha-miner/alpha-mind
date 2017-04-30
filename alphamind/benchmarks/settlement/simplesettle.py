# -*- coding: utf-8 -*-
"""
Created on 2017-4-28

@author: cheng.li
"""

import datetime as dt
import numpy as np
import pandas as pd
from alphamind.settlement.simplesettle import simple_settle


def benchmark_simple_settle(n_samples: int, n_portfolios: int, n_loops: int) -> None:
    print("-" * 60)
    print("Starting simple settle benchmarking")
    print("Parameters(n_samples: {0}, n_portfolios: {1}, n_loops: {2})".format(n_samples, n_portfolios, n_loops))

    weights = np.random.randn(n_samples, n_portfolios)
    ret_series = np.random.randn(n_samples)

    start = dt.datetime.now()
    for _ in range(n_loops):
        calc_ret = simple_settle(weights, ret_series)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    ret_series.shape = -1, 1
    for _ in range(n_loops):
        exp_ret = (weights * ret_series).sum(axis=0)
    benchmark_model_time = dt.datetime.now() - start

    np.testing.assert_array_almost_equal(calc_ret, exp_ret)

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


def benchmark_simple_settle_with_group(n_samples: int, n_portfolios: int, n_loops: int, n_groups: int) -> None:
    print("-" * 60)
    print("Starting simple settle with group-by values benchmarking")
    print("Parameters(n_samples: {0}, n_portfolios: {1}, n_loops: {2}, n_groups: {3})".format(n_samples, n_portfolios, n_loops, n_groups))

    weights = np.random.randn(n_samples, n_portfolios)
    ret_series = np.random.randn(n_samples)
    groups = np.random.randint(n_groups, size=n_samples)

    start = dt.datetime.now()
    for _ in range(n_loops):
        calc_ret = simple_settle(weights, ret_series, groups=groups)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    ret_series.shape = -1, 1
    for _ in range(n_loops):
        ret_mat = weights * ret_series
        exp_ret = pd.DataFrame(ret_mat).groupby(groups, sort=False).sum().values
    benchmark_model_time = dt.datetime.now() - start

    np.testing.assert_array_almost_equal(calc_ret, exp_ret)

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


if __name__ == '__main__':
    benchmark_simple_settle(3000, 3, 1000)
    benchmark_simple_settle_with_group(3000, 3, 1000, 30)
