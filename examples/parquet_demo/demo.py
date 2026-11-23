from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from alphamind.analysis.perfanalysis import aggregate_attribution
from alphamind.analysis.quantileanalysis import quantile_analysis
from alphamind.analysis.riskanalysis import risk_analysis
from alphamind.data.arrow import column_values, read_parquet, to_numpy, with_column, write_parquet
from alphamind.data.calendar import advanceDateByCalendar, makeSchedule
from alphamind.data.processing import factor_processing
from alphamind.data.standardize import standardize
from alphamind.data.transformer import Transformer
from alphamind.data.winsorize import winsorize_normal
from alphamind.execution.naiveexecutor import NaiveExecutor
from alphamind.execution.pipeline import ExecutionPipeline
from alphamind.execution.thresholdexecutor import ThresholdExecutor
from alphamind.formula import MA, SHIFT, CSQuantiles, CSRank
from alphamind.model.composer import predict_by_model, train_model
from alphamind.model.linearmodel import LinearRegression
from alphamind.portfolio.evolver import evolve_positions
from alphamind.portfolio.meanvariancebuilder import mean_variance_builder
from alphamind.portfolio.rankbuilder import rank_build
from alphamind.portfolio.riskmodel import FactorRiskModel
from alphamind.settlement.simplesettle import simple_settle

FEATURES = ["momentum", "value"]
N_ASSETS = 300
N_BACKTEST_DAYS = 252
WARMUP_DAYS = 2
N_SELECTED = 50
CODES = np.array([f"{index:06d}" for index in range(1, N_ASSETS + 1)])


def save(table: pa.Table, output: Path, relative_path: str) -> Path:
    return write_parquet(table, output / relative_path)


def sections(table: pa.Table):
    ordered = table.sort_by([("trade_date", "ascending"), ("code", "ascending")])
    if ordered.num_rows == 0:
        return
    dates = ordered["trade_date"].combine_chunks()
    values = pc.cast(dates, pa.int64()).to_numpy(zero_copy_only=False)
    starts = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1))
    ends = np.concatenate((starts[1:], [ordered.num_rows]))
    for start, end in zip(starts, ends):
        yield dates[start].as_py(), ordered.slice(start, end - start)


def create_inputs(output: Path) -> pa.Table:
    start_date = "2026-01-05"
    end_date = advanceDateByCalendar(
        "china.sse", start_date, f"{N_BACKTEST_DAYS + WARMUP_DAYS - 1}b"
    )
    dates = makeSchedule(start_date, end_date.strftime("%Y-%m-%d"), "1b", "china.sse")
    n_dates = len(dates)
    n_assets = len(CODES)
    rng = np.random.default_rng(20261123)
    asset_index = np.arange(n_assets)
    industries = np.tile(np.array(["Bank", "Auto", "Health"]), n_assets // 3)
    beta = np.tile(1.0 + 0.2 * np.linspace(-1.0, 1.0, n_assets), n_dates)
    size = np.tile(np.linspace(-1.0, 1.0, n_assets), n_dates)
    returns = rng.normal(0.001, 0.015, size=(n_dates, n_assets))
    close = (10.0 + asset_index)[None, :] * np.cumprod(1.0 + returns, axis=0)
    forward_returns = np.vstack([returns[1:], returns[-1:]])
    benchmark = np.linspace(1.0, 2.0, n_assets)
    benchmark /= benchmark.sum()
    aware_dates = [value.replace(tzinfo=dt.timezone.utc) for value in dates]
    research = pa.table(
        {
            "trade_date": [value for value in aware_dates for _ in CODES],
            "code": np.tile(CODES, n_dates),
            "close": close.reshape(-1),
            "raw_momentum": 0.4 * beta - 0.3 * size + rng.normal(size=n_dates * n_assets),
            "raw_value": -0.2 * beta + 0.5 * size + rng.normal(size=n_dates * n_assets),
            "dx": np.log1p(forward_returns).reshape(-1),
            "industry": np.tile(industries, n_dates),
            "benchmark_weight": np.tile(benchmark, n_dates),
            "BETA": beta,
            "SIZE": size + rng.normal(0.0, 0.01, size=n_dates * n_assets),
        }
    ).sort_by([("trade_date", "ascending"), ("code", "ascending")])
    save(research, output, "input/research.parquet")
    save(
        research.select(["trade_date", "code", "BETA", "SIZE"]),
        output,
        "input/risk_exposure.parquet",
    )
    factor_covariance = pa.table(
        {
            "Factor": ["BETA", "SIZE"],
            "BETA": [0.0004, 0.00002],
            "SIZE": [0.00002, 0.0009],
        }
    )
    save(factor_covariance, output, "input/factor_covariance.parquet")
    latest_date = aware_dates[-1]
    latest = research.filter(pc.equal(research["trade_date"], latest_date))
    specific_risk = pa.table(
        {
            "code": latest["code"],
            "idsync": np.linspace(0.0003, 0.0008, n_assets),
        }
    )
    save(specific_risk, output, "input/specific_risk.parquet")
    schedule = pa.table({"trade_date": aware_dates, "sequence": np.arange(1, n_dates + 1)})
    save(schedule, output, "calendar/schedule.parquet")
    return research


def calculate_features(output: Path) -> pa.Table:
    research = read_parquet(output / "input/research.parquet")
    transformer = Transformer(
        {
            "momentum": CSRank("raw_momentum"),
            "value": CSQuantiles("raw_value"),
            "close_ma_3": MA(3, "close"),
            "close_lag_1": SHIFT("close", 1),
        }
    )
    transformed = transformer.transform("code", research)
    features = research
    for name in transformer.names:
        features = features.append_column(name, transformed[name])
    features = features.filter(pc.is_valid(features["close_ma_3"]))
    save(
        features.select(
            [
                "trade_date",
                "code",
                "momentum",
                "value",
                "close_ma_3",
                "close_lag_1",
                "SIZE",
                "dx",
            ]
        ),
        output,
        "features/calculated.parquet",
    )
    return features


def process_features(output: Path) -> pa.Table:
    features = read_parquet(output / "features/calculated.parquet")
    processed_sections = []
    for _, section in sections(features):
        values = to_numpy(section, FEATURES, dtype=float)
        risks = to_numpy(section, ["SIZE"], dtype=float)
        processed = factor_processing(
            values,
            pre_process=[winsorize_normal, standardize],
            risk_factors=risks,
            post_process=[standardize],
        )
        result = section
        for index, name in enumerate(FEATURES):
            result = with_column(result, name, processed[:, index])
        processed_sections.append(result)
    processed = pa.concat_tables(processed_sections)
    save(
        processed.select(["trade_date", "code", "dx"] + FEATURES),
        output,
        "features/processed.parquet",
    )
    return processed


def train_and_predict(output: Path) -> pa.Table:
    processed = read_parquet(
        output / "features/processed.parquet",
        columns=["trade_date", "code", "dx"] + FEATURES,
    )
    dates = sorted(pc.unique(processed["trade_date"]).to_pylist())
    latest_date = dates[-1]
    train = processed.filter(pc.not_equal(processed["trade_date"], latest_date))
    latest = processed.filter(pc.equal(processed["trade_date"], latest_date)).sort_by("code")
    model, _, _ = train_model(
        latest_date.strftime("%Y-%m-%d"),
        LinearRegression(features=FEATURES, fit_intercept=True),
        x_values=train,
        y_values=column_values(train, "dx"),
    )
    predictions, _ = predict_by_model(
        latest_date.strftime("%Y-%m-%d"),
        model,
        x_values=latest,
        codes=latest["code"].to_pylist(),
    )
    predictions = predictions.append_column(
        "trade_date", pa.array([latest_date] * predictions.num_rows)
    )
    save(predictions, output, "model/predictions.parquet")
    return predictions


def run_file_backtest(output: Path) -> pa.Table:
    processed = read_parquet(
        output / "features/processed.parquet",
        columns=["trade_date", "code", "dx"] + FEATURES,
    )
    pipeline = ExecutionPipeline([ThresholdExecutor(turn_over_threshold=0.02, is_relative=False)])
    positions = []
    performance = []
    for trade_date, section in sections(processed):
        expected_return = to_numpy(section, FEATURES, dtype=float) @ np.array([0.6, 0.4])
        weights = rank_build(expected_return, use_rank=N_SELECTED).flatten().astype(float)
        weights /= weights.sum()
        target = pa.table(
            {
                "trade_date": [trade_date] * section.num_rows,
                "code": section["code"],
                "weight": weights,
            }
        )
        turn_over, executed = pipeline.execute(target)
        executed_weights = column_values(executed, "weight")
        portfolio_return = float(executed_weights @ column_values(section, "dx"))
        performance.append(
            pa.table(
                {
                    "trade_date": [trade_date],
                    "return": [portfolio_return],
                    "turn_over": [turn_over],
                    "leverage": [float(np.abs(executed_weights).sum())],
                }
            )
        )
        positions.append(executed)
        pipeline.update({"return": portfolio_return})
    positions_table = pa.concat_tables(positions)
    performance_table = pa.concat_tables(performance)
    turnover = performance_table.select(["trade_date", "turn_over"]).rename_columns(
        ["trade_date", "weight"]
    )
    save(positions_table, output, "backtest/positions.parquet")
    save(performance_table, output, "backtest/returns.parquet")
    save(turnover, output, "backtest/turnover.parquet")
    return positions_table


def optimize_latest(output: Path) -> pa.Table:
    dates = read_parquet(output / "features/processed.parquet", columns=["trade_date"])
    latest_date = pc.max(dates["trade_date"]).as_py()
    filters = [("trade_date", "=", latest_date)]
    exposure = read_parquet(output / "input/risk_exposure.parquet", filters=filters).sort_by("code")
    predictions = read_parquet(output / "model/predictions.parquet", filters=filters).sort_by(
        "code"
    )
    latest = read_parquet(
        output / "input/research.parquet",
        columns=["trade_date", "code", "benchmark_weight", "industry", "dx"],
        filters=filters,
    ).sort_by("code")
    covariance = read_parquet(output / "input/factor_covariance.parquet")
    specific = read_parquet(output / "input/specific_risk.parquet")
    risk_model = FactorRiskModel(covariance, exposure, specific)
    codes = exposure["code"].to_pylist()
    benchmark = column_values(latest, "benchmark_weight")
    size = column_values(exposure, "SIZE")
    total_and_size = np.column_stack([np.ones(len(codes)), size])
    size_target = float(benchmark @ size)
    risk_target = (np.array([1.0, size_target]), np.array([1.0, size_target]))
    status, _, weights = mean_variance_builder(
        column_values(predictions, "prediction"),
        risk_model.get_risk_profile(codes),
        benchmark,
        np.zeros(len(codes)),
        np.full(len(codes), 0.2),
        total_and_size,
        risk_target,
        lam=2.0,
    )
    if status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Portfolio optimization failed: {status}")
    target = pa.table(
        {
            "trade_date": [latest_date] * len(codes),
            "code": codes,
            "weight": weights,
            "benchmark_weight": benchmark,
            "industry": latest["industry"],
            "dx": latest["dx"],
            "BETA": exposure["BETA"],
            "SIZE": exposure["SIZE"],
        }
    )
    turn_over, executed = NaiveExecutor().execute(target)
    save(target, output, "portfolio/optimized.parquet")
    save(executed, output, "portfolio/executed.parquet")
    save(
        pa.table({"trade_date": [latest_date], "turn_over": [turn_over]}),
        output,
        "portfolio/execution.parquet",
    )
    evolved = with_column(
        target,
        "weight",
        evolve_positions(weights, column_values(target, "dx")),
    )
    save(evolved, output, "portfolio/evolved.parquet")
    return executed


def analyze_results(output: Path) -> None:
    executed = read_parquet(output / "portfolio/executed.parquet")
    latest_date = executed["trade_date"][0].as_py()
    latest = read_parquet(
        output / "features/processed.parquet",
        columns=["trade_date", "code", "dx"] + FEATURES,
        filters=[("trade_date", "=", latest_date)],
    ).sort_by("code")
    quantiles = quantile_analysis(
        latest.select(FEATURES),
        np.array([0.6, 0.4]),
        column_values(latest, "dx"),
        n_bins=4,
        pre_process=[],
        post_process=[],
    )
    save(
        pa.table({"quantile": np.arange(1, 5), "return": quantiles}),
        output,
        "analysis/quantile_returns.parquet",
    )
    weights = column_values(executed, "weight")
    benchmark = column_values(executed, "benchmark_weight")
    settlement = simple_settle(
        weights,
        column_values(executed, "dx"),
        groups=executed["industry"].to_pylist(),
        benchmark=benchmark,
    )
    save(settlement, output, "analysis/settlement.parquet")
    active = with_column(executed, "weight", weights - benchmark).select(
        ["trade_date", "code", "weight"]
    )
    returns = pa.table(
        {
            "trade_date": executed["trade_date"],
            "code": executed["code"],
            "return": executed["dx"],
        }
    )
    risks = executed.select(["trade_date", "code", "BETA", "SIZE"])
    explained, exposure = risk_analysis(active, returns, risks)
    attribution = aggregate_attribution(explained)
    save(explained, output, "analysis/security_attribution.parquet")
    save(exposure, output, "analysis/risk_contributions.parquet")
    save(attribution, output, "analysis/return_attribution.parquet")


def write_manifest(output: Path) -> pa.Table:
    artifacts = [
        path for path in sorted(output.rglob("*.parquet")) if path.name != "manifest.parquet"
    ]
    manifest = pa.table(
        {
            "artifact": [str(path.relative_to(output)) for path in artifacts],
            "rows": [pq.ParquetFile(path).metadata.num_rows for path in artifacts],
        }
    )
    save(manifest, output, "manifest.parquet")
    return manifest


def run_demo(output: Path) -> pa.Table:
    output.mkdir(parents=True, exist_ok=True)
    create_inputs(output)
    calculate_features(output)
    process_features(output)
    train_and_predict(output)
    run_file_backtest(output)
    optimize_latest(output)
    analyze_results(output)
    return write_manifest(output)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Arrow-native Parquet research demo")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("output"),
        help="Directory for generated Parquet files",
    )
    return parser


def main(argv=None) -> int:
    args = get_parser().parse_args(argv)
    manifest = run_demo(args.output)
    print(f"Created {manifest.num_rows} Parquet artifacts in {args.output.resolve()}")
    print(manifest.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
