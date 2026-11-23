# -*- coding: utf-8 -*-
"""
Created on 2017-4-28

@author: cheng.li
"""

import datetime as dt

import numpy as np

from alphamind.settlement.simplesettle import simple_settle


def _settle_portfolios(weights, ret_series, groups=None):
    return np.array(
        [
            simple_settle(weights[:, index], ret_series, groups)["er"].to_pylist()
            for index in range(weights.shape[1])
        ]
    )


def benchmark_simple_settle(n_samples: int, n_portfolios: int, n_loops: int) -> None:
    print("-" * 60)
    print("Starting simple settle benchmarking")
    print(
        "Parameters(n_samples: {0}, n_portfolios: {1}, n_loops: {2})".format(
            n_samples, n_portfolios, n_loops
        )
    )

    weights = np.random.randn(n_samples, n_portfolios)
    ret_series = np.random.randn(n_samples)

    start = dt.datetime.now()
    for _ in range(n_loops):
        calc_ret = _settle_portfolios(weights, ret_series).ravel()
    impl_model_time = dt.datetime.now() - start

    print("{0:20s}: {1}".format("Implemented model", impl_model_time))

    start = dt.datetime.now()
    ret_series = np.reshape(ret_series, (-1, 1))
    for _ in range(n_loops):
        exp_ret = (weights * ret_series).sum(axis=0)
    benchmark_model_time = dt.datetime.now() - start

    np.testing.assert_array_almost_equal(calc_ret, exp_ret)

    print("{0:20s}: {1}".format("Benchmark model", benchmark_model_time))


def benchmark_simple_settle_with_group(
    n_samples: int, n_portfolios: int, n_loops: int, n_groups: int
) -> None:
    print("-" * 60)
    print("Starting simple settle with group-by values benchmarking")
    print(
        "Parameters(n_samples: {0}, n_portfolios: {1}, n_loops: {2}, n_groups: {3})".format(
            n_samples, n_portfolios, n_loops, n_groups
        )
    )

    weights = np.random.randn(n_samples, n_portfolios)
    ret_series = np.random.randn(n_samples)
    groups = np.random.randint(n_groups, size=n_samples)

    start = dt.datetime.now()
    for _ in range(n_loops):
        calc_ret = _settle_portfolios(weights, ret_series, groups=groups)
    impl_model_time = dt.datetime.now() - start

    print("{0:20s}: {1}".format("Implemented model", impl_model_time))

    start = dt.datetime.now()
    ret_series = np.reshape(ret_series, (-1, 1))
    for _ in range(n_loops):
        ret_mat = weights * ret_series
        exp_ret = np.array([ret_mat[groups == group].sum(axis=0) for group in np.unique(groups)])
        exp_ret = np.vstack([exp_ret, exp_ret.sum(axis=0)])
    benchmark_model_time = dt.datetime.now() - start

    np.testing.assert_array_almost_equal(calc_ret.T, exp_ret)

    print("{0:20s}: {1}".format("Benchmark model", benchmark_model_time))


if __name__ == "__main__":
    benchmark_simple_settle(3000, 3, 1000)
    benchmark_simple_settle_with_group(3000, 3, 1000, 30)
