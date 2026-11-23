from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa

from alphamind.analysis.calculators import calculate_turn_over
from alphamind.data.arrow import read_parquet, write_parquet
from alphamind.data.neutralize import neutralize
from alphamind.data.standardize import standardize
from alphamind.data.transformer import Transformer
from alphamind.data.winsorize import winsorize_normal
from alphamind.formula import MA, CSRank
from alphamind.portfolio.percentbuilder import percent_build
from alphamind.portfolio.rankbuilder import rank_build


def _matrix_operation(size, operation):
    values = np.random.default_rng(42).normal(size=(size, 4))
    return lambda: operation(values)


def _neutralize_operation(size):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(size, 2))
    risks = rng.normal(size=(size, 4))
    return lambda: neutralize(risks, values)


def _rank_operation(size):
    values = np.random.default_rng(42).normal(size=(size, 4))
    return lambda: rank_build(values, max(1, size // 10))


def _percent_operation(size):
    values = np.random.default_rng(42).normal(size=(size, 4))
    return lambda: percent_build(values, 0.1)


def _calc_flow_operation(size):
    half = (size + 1) // 2
    dates = [dt.datetime(2020, 1, 1) + dt.timedelta(days=index) for index in range(half)]
    table = pa.table(
        {
            "trade_date": [value for value in dates for _ in range(2)][:size],
            "code": (["A", "B"] * half)[:size],
            "x": np.random.default_rng(42).normal(size=size),
        }
    )
    transformer = Transformer({"mean": MA(5, "x"), "rank": CSRank("x")})
    return lambda: transformer.transform("code", table)


def _turnover_operation(size):
    if size >= 300 and size % 300 == 0:
        n_assets = 300
    elif size >= 100 and size % 100 == 0:
        n_assets = 100
    else:
        n_assets = 1
    n_dates = size // n_assets
    dates = [
        dt.datetime(2020, 1, 1) + dt.timedelta(days=index)
        for index in range(n_dates)
        for _ in range(n_assets)
    ]
    codes = [f"{index:06d}" for index in range(n_assets)] * n_dates
    table = pa.table(
        {
            "trade_date": dates,
            "code": codes,
            "weight": np.random.default_rng(42).normal(size=size) / n_assets,
        }
    )
    return lambda: calculate_turn_over(table)


def _parquet_operation(size):
    table = pa.table(
        {
            "code": [f"{index:06d}" for index in range(size)],
            "value": np.random.default_rng(42).normal(size=size),
        }
    )

    def run():
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.parquet"
            write_parquet(table, path)
            return read_parquet(path)

    return run


WORKLOADS: dict[str, Callable[[int], Callable[[], object]]] = {
    "winsorize": lambda size: _matrix_operation(size, winsorize_normal),
    "standardize": lambda size: _matrix_operation(size, standardize),
    "neutralize": _neutralize_operation,
    "rank_build": _rank_operation,
    "percent_build": _percent_operation,
    "calc_flow": _calc_flow_operation,
    "turnover": _turnover_operation,
    "parquet": _parquet_operation,
}


def run_case(name: str, size: int, samples: int = 10, warmup: int = 3) -> dict:
    if size < 1:
        raise ValueError("size must be positive")
    if samples < 1:
        raise ValueError("samples must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    operation = WORKLOADS[name](size)
    for _ in range(warmup):
        operation()
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        operation()
        timings.append((time.perf_counter_ns() - started) / 1_000_000_000)
    median = statistics.median(timings)
    return {
        "name": name,
        "size": size,
        "samples": samples,
        "warmup": warmup,
        "seconds": timings,
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "mean_seconds": statistics.mean(timings),
        "median_seconds": median,
        "rows_per_second": size / median,
    }


def _package_version():
    try:
        return version("alpha-mind")
    except PackageNotFoundError:
        return "unknown"


def _summary(report):
    lines = [
        "# Alpha Mind benchmarks",
        "",
        f"Commit: `{report['git_sha']}`; Python: `{report['python_version']}`; platform: `{report['platform']}`.",
        "",
        "| Case | Rows | Samples | Median (ms) | Rows / second |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        lines.append(
            f"| {result['name']} | {result['size']} | {result['samples']} "
            f"| {result['median_seconds'] * 1000:.3f} | {result['rows_per_second']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def run_suite(
    output: Path,
    size: int = 10_000,
    samples: int = 10,
    warmup: int = 3,
    cases=None,
) -> dict:
    import os

    selected = list(cases) if cases else list(WORKLOADS)
    report = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "alpha_mind_version": _package_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "results": [run_case(name, size, samples, warmup) for name in selected],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(_summary(report), encoding="utf-8")
    return report


def get_parser():
    parser = argparse.ArgumentParser(description="Run reproducible Alpha Mind benchmarks")
    parser.add_argument("--case", action="append", choices=sorted(WORKLOADS), dest="cases")
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    run_suite(args.output, args.size, args.samples, args.warmup, args.cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
