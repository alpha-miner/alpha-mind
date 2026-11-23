import numpy as np
import pytest

from alphamind.benchmarks.data.neutralize import (
    benchmark_neutralize,
    benchmark_neutralize_with_groups,
)
from alphamind.benchmarks.data.standardize import (
    benchmark_standardize,
    benchmark_standardize_with_group,
)
from alphamind.benchmarks.data.winsorize import (
    benchmark_winsorize_normal,
    benchmark_winsorize_normal_with_group,
)
from alphamind.benchmarks.portfolio.linearbuild import benchmark_build_linear
from alphamind.benchmarks.portfolio.percentbuild import (
    benchmark_build_percent,
    benchmark_build_percent_with_group,
)
from alphamind.benchmarks.portfolio.rankbuild import (
    benchmark_build_rank,
    benchmark_build_rank_with_group,
)
from alphamind.benchmarks.settlement.simplesettle import (
    benchmark_simple_settle,
    benchmark_simple_settle_with_group,
)


@pytest.fixture(autouse=True)
def deterministic_random_data():
    np.random.seed(7)


@pytest.mark.parametrize(
    "benchmark",
    [
        lambda: benchmark_neutralize(20, 3, 1),
        lambda: benchmark_neutralize_with_groups(20, 3, 1, 2),
        lambda: benchmark_standardize(20, 3, 1),
        lambda: benchmark_standardize_with_group(20, 3, 1, 2),
        lambda: benchmark_winsorize_normal(20, 3, 1),
        lambda: benchmark_winsorize_normal_with_group(20, 3, 1, 2),
        lambda: benchmark_build_rank(20, 5, 3),
        lambda: benchmark_build_rank_with_group(20, 5, 3, 2),
        lambda: benchmark_build_percent(20, 5, 0.2),
        lambda: benchmark_build_percent_with_group(20, 5, 0.2, 2),
        lambda: benchmark_build_linear(50, 3, 2),
        lambda: benchmark_simple_settle(20, 3, 5),
        lambda: benchmark_simple_settle_with_group(20, 3, 5, 2),
    ],
)
def test_benchmark_reference_implementations(benchmark, capsys):
    benchmark()
    output = capsys.readouterr().out
    assert "Implemented model" in output
    assert "Benchmark model" in output
