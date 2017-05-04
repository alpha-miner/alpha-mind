# -*- coding: utf-8 -*-
"""
Created on 2017-5-4

@author: cheng.li
"""

import datetime as dt
import numpy as np
import pandas as pd
from alphamind.portfolio.percentbuilder import percent_build


def benchmark_build_percent(n_samples: int, n_loops: int, p_included: float) -> None:
    print("-" * 60)
    print("Starting portfolio construction by percent benchmarking")
    print("Parameters(n_samples: {0}, p_included: {1}, n_loops: {2})".format(n_samples, p_included, n_loops))

    n_portfolio = 10

    x = np.random.randn(n_samples, n_portfolio)

    start = dt.datetime.now()
    for _ in range(n_loops):
        calc_weights = percent_build(x, p_included)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    for _ in range(n_loops):
        exp_weights = np.zeros((len(x), n_portfolio))
        n_incuded = int(p_included * len(x))
        choosed_index = (-x).argsort(axis=0).argsort(axis=0) < n_incuded
        for j in range(n_portfolio):
            exp_weights[choosed_index[:, j], j] = 1.
    benchmark_model_time = dt.datetime.now() - start

    np.testing.assert_array_almost_equal(calc_weights, exp_weights)

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


def benchmark_build_percent_with_group(n_samples: int, n_loops: int, p_included: float, n_groups: int) -> None:
    print("-" * 60)
    print("Starting  portfolio construction by percent with group-by values benchmarking")
    print("Parameters(n_samples: {0}, p_included: {1}, n_loops: {2}, n_groups: {3})".format(n_samples, p_included, n_loops, n_groups))

    n_portfolio = 10

    x = np.random.randn(n_samples, n_portfolio)
    groups = np.random.randint(n_groups, size=n_samples)

    start = dt.datetime.now()
    for _ in range(n_loops):
        calc_weights = percent_build(x, p_included, groups=groups)
    impl_model_time = dt.datetime.now() - start

    print('{0:20s}: {1}'.format('Implemented model', impl_model_time))

    start = dt.datetime.now()
    for _ in range(n_loops):
        grouped_ordering = pd.DataFrame(-x).groupby(groups).rank()
        grouped_count = pd.DataFrame(-x).groupby(groups).transform(lambda x: x.count())
        exp_weights = np.zeros((len(x), n_portfolio))
        n_included = (grouped_count * p_included).astype(int)
        masks = (grouped_ordering <= n_included).values
        for j in range(n_portfolio):
            exp_weights[masks[:, j], j] = 1.
    benchmark_model_time = dt.datetime.now() - start

    np.testing.assert_array_almost_equal(calc_weights, exp_weights)

    print('{0:20s}: {1}'.format('Benchmark model', benchmark_model_time))


if __name__ == '__main__':
    benchmark_build_percent(3000, 1000, 0.1)
    benchmark_build_percent_with_group(3000, 1000, 0.1, 30)
