from __future__ import annotations

import bisect
from typing import Iterable, Union

import calc_flow as cf
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from alphamind.data.arrow import (
    as_arrow,
    column_values,
    drop_nulls,
    from_numpy,
    timestamp_scalar,
    to_numpy,
    with_column,
)
from alphamind.data.calendar import (
    BizDayConventions,
    DateGeneration,
    Period,
    advanceDateByCalendar,
    makeSchedule,
)
from alphamind.data.engines.sqlengine import SqlEngine, total_risk_factors
from alphamind.data.engines.universe import Universe
from alphamind.data.processing import factor_processing
from alphamind.data.transformer import Transformer
from alphamind.utilities import map_freq


def _ordered_input(table):
    schema = pa.schema(
        [
            pa.field(
                field.name,
                field.type,
                nullable=False if field.name in ("trade_date", "code") else field.nullable,
            )
            for field in table.schema
        ]
    )
    table = pa.Table.from_arrays(table.columns, schema=schema)
    source = cf.table_input(
        "input",
        schema=schema,
        entity_by=("code",),
        event_time="trade_date",
        sequence_by=("trade_date",),
    )
    return table, source


def _forward_fill(table, columns):
    table = as_arrow(table).sort_by([("code", "ascending"), ("trade_date", "ascending")])
    table, source = _ordered_input(table)
    temporary = {f"__fill_{index}": name for index, name in enumerate(columns)}
    result = source.with_columns(
        {temporary_name: cf.ts.last(source[name]) for temporary_name, name in temporary.items()}
    ).collect(table)
    for temporary_name, name in temporary.items():
        result = with_column(result, name, result[temporary_name])
    return result.drop(list(temporary))


def _median_fill(table, columns):
    table = as_arrow(table)
    projections = []
    for name in table.column_names:
        quoted = name.replace('"', '""')
        if name in columns:
            projections.append(
                f'COALESCE("{quoted}", MEDIAN("{quoted}") OVER (PARTITION BY trade_date)) '
                f'AS "{quoted}"'
            )
        else:
            projections.append(f'"{quoted}"')
    query = "SELECT " + ", ".join(projections) + " FROM input"
    return cf.compute(table, lambda input_table: input_table.sql(query))


def _merge_df(engine, names, factor_df, target_df, universe, dates, risk_model, neutralized_risk):
    risk_df = engine.fetch_risk_model_range(universe, dates=dates, risk_model=risk_model)[1]
    used_risks = sorted(set(total_risk_factors).difference(names))
    risk_df = drop_nulls(risk_df.select(["trade_date", "code"] + used_risks))
    target_df = drop_nulls(target_df.join(risk_df, keys=["trade_date", "code"], join_type="inner"))
    if neutralized_risk:
        train_x = factor_df.join(risk_df, keys=["trade_date", "code"], join_type="inner")
        train_y = target_df
        risk_exp = to_numpy(train_x, neutralized_risk, dtype=float)
    else:
        train_x = factor_df
        train_y = target_df
        risk_exp = None
    order = [("trade_date", "ascending"), ("code", "ascending")]
    train_x = train_x.sort_by(order)
    train_y = train_y.sort_by(order)
    target_df = target_df.sort_by(order)
    x_values = to_numpy(train_x, names, dtype=float)
    y_values = to_numpy(train_y, ["dx"], dtype=float)
    codes = column_values(train_x, "code")
    date_label = np.asarray(train_x["trade_date"].to_pylist())
    dates = np.unique(date_label)
    return target_df, dates, date_label, risk_exp, x_values, y_values, train_x, train_y, codes


def _fit_target(engine, universe, dates, fit_target, horizon, frequency):
    if fit_target is None:
        return engine.fetch_dx_return_range(universe, dates=dates, horizon=horizon)
    one_more = advanceDateByCalendar("china.sse", dates[-1], frequency).strftime("%Y-%m-%d")
    target = engine.fetch_factor_range_forward(
        universe, factors=fit_target, dates=list(dates) + [one_more]
    )
    values = pa.array(
        [timestamp_scalar(value).value for value in dates], type=pa.timestamp("us", tz="UTC")
    )
    target = target.filter(pc.is_in(target["trade_date"], value_set=values))
    return _forward_fill(target, ["dx"])


def prepare_data(
    engine: SqlEngine,
    factors: Union[Transformer, Iterable[object]],
    start_date: str,
    end_date: str,
    frequency: str,
    universe: Universe,
    benchmark: int,
    warm_start: int = 0,
    fit_target: Union[Transformer, object] = None,
):
    if warm_start > 0:
        period = Period(frequency)
        period = Period(length=-warm_start * period.length(), units=period.units())
        start_date = advanceDateByCalendar("china.sse", start_date, period).strftime("%Y-%m-%d")
    dates = [
        value.strftime("%Y-%m-%d")
        for value in makeSchedule(
            start_date,
            end_date,
            frequency,
            calendar="china.sse",
            dateRule=BizDayConventions.Following,
            dateGenerationRule=DateGeneration.Forward,
        )
    ]
    horizon = map_freq(frequency)
    transformer = factors if isinstance(factors, Transformer) else Transformer(factors)
    factor_df = engine.fetch_factor_range(universe, factors=transformer, dates=dates)
    target_df = _fit_target(engine, universe, dates, fit_target, horizon, frequency)
    industry_df = engine.fetch_industry_range(universe, dates=dates)
    benchmark_df = engine.fetch_benchmark_range(benchmark, dates=dates)
    keys = ["trade_date", "code"]
    data = drop_nulls(factor_df.join(target_df, keys=keys, join_type="inner"))
    data = data.join(benchmark_df, keys=keys, join_type="left outer")
    data = with_column(data, "weight", pc.fill_null(data["weight"], 0.0))
    data = drop_nulls(data.join(industry_df, keys=keys, join_type="inner"))
    return (
        dates,
        data.select(keys + ["dx"]),
        data.select(keys + ["weight", "industry_code", "industry"] + transformer.names),
    )


def batch_processing(
    names,
    x_values,
    y_values,
    groups,
    group_label,
    batch,
    risk_exp,
    pre_process,
    post_process,
    codes,
):
    train_x_buckets, train_y_buckets, train_risk_buckets = {}, {}, {}
    predict_x_buckets, predict_y_buckets, predict_risk_buckets = {}, {}, {}
    predict_codes_bucket = {}
    for index, start in enumerate(groups[:-batch]):
        end = groups[index + batch]
        left = bisect.bisect_left(group_label, start)
        right = bisect.bisect_left(group_label, end)
        raw_x = x_values[left:right]
        raw_y = y_values[left:right]
        risks = risk_exp[left:right] if risk_exp is not None else None
        train_x_buckets[end] = from_numpy(
            factor_processing(
                raw_x, pre_process=pre_process, risk_factors=risks, post_process=post_process
            ),
            names,
        )
        train_y_buckets[end] = factor_processing(
            raw_y, pre_process=pre_process, risk_factors=risks, post_process=post_process
        )
        train_risk_buckets[end] = risks
        left = bisect.bisect_right(group_label, start)
        right = bisect.bisect_right(group_label, end)
        sub_dates = group_label[left:right]
        raw_x = x_values[left:right]
        sub_codes = codes[left:right]
        risks = risk_exp[left:right] if risk_exp is not None else None
        processed = factor_processing(
            raw_x, pre_process=pre_process, risk_factors=risks, post_process=post_process
        )
        inner_left = bisect.bisect_left(sub_dates, end)
        inner_right = bisect.bisect_right(sub_dates, end)
        predict_x_buckets[end] = from_numpy(processed[inner_left:inner_right], names)
        if risks is not None:
            predict_risk_buckets[end] = risks[inner_left:inner_right]
        predict_codes_bucket[end] = sub_codes[inner_left:inner_right]
        raw_y = y_values[left:right]
        if len(raw_y) > 0:
            processed_y = factor_processing(
                raw_y, pre_process=pre_process, risk_factors=risks, post_process=post_process
            )
            predict_y_buckets[end] = processed_y[inner_left:inner_right]
    return (
        train_x_buckets,
        train_y_buckets,
        train_risk_buckets,
        predict_x_buckets,
        predict_y_buckets,
        predict_risk_buckets if risk_exp is not None else None,
        predict_codes_bucket,
    )


def fetch_data_package(
    engine: SqlEngine,
    alpha_factors: Iterable[object],
    start_date: str,
    end_date: str,
    frequency: str,
    universe: Universe,
    benchmark: int,
    warm_start: int = 0,
    batch: int = 1,
    neutralized_risk: Iterable[str] = None,
    risk_model: str = "short",
    pre_process: Iterable[object] = None,
    post_process: Iterable[object] = None,
    fit_target: Union[Transformer, object] = None,
) -> dict:
    transformer = Transformer(alpha_factors)
    dates, target_df, factor_df = prepare_data(
        engine,
        transformer,
        start_date,
        end_date,
        frequency,
        universe,
        benchmark,
        warm_start + batch,
        fit_target=fit_target,
    )
    target_df, dates, date_label, risk_exp, x_values, y_values, train_x, _, codes = _merge_df(
        engine,
        transformer.names,
        factor_df,
        target_df,
        universe,
        dates,
        risk_model,
        neutralized_risk,
    )
    for name in ("weight", "industry", "industry_code"):
        target_df = with_column(target_df, name, train_x[name])
    if neutralized_risk:
        for index, name in enumerate(neutralized_risk):
            target_df = with_column(target_df, name, risk_exp[:, index])
    buckets = batch_processing(
        transformer.names,
        x_values,
        y_values,
        dates,
        date_label,
        batch,
        risk_exp,
        pre_process,
        post_process,
        codes,
    )
    (train_x, train_y, train_risk, predict_x, predict_y, predict_risk, predict_codes) = buckets

    def recent(mapping):
        return {
            key: value for key, value in mapping.items() if key.strftime("%Y-%m-%d") >= start_date
        }

    train_x, train_y, train_risk = recent(train_x), recent(train_y), recent(train_risk)
    predict_x, predict_y, predict_codes = (
        recent(predict_x),
        recent(predict_y),
        recent(predict_codes),
    )
    predict_risk = recent(predict_risk) if predict_risk is not None else None
    start_timestamp = timestamp_scalar(start_date)
    return {
        "x_names": transformer.names,
        "settlement": target_df.filter(pc.greater_equal(target_df["trade_date"], start_timestamp)),
        "train": {"x": train_x, "y": train_y, "risk": train_risk},
        "predict": {"x": predict_x, "y": predict_y, "risk": predict_risk, "code": predict_codes},
    }


def _training_window(
    engine, transformer, ref_date, frequency, universe, batch, warm_start, fit_target
):
    period = Period(frequency)
    period = Period(length=-(warm_start + batch) * period.length(), units=period.units())
    start = advanceDateByCalendar("china.sse", ref_date, period, BizDayConventions.Following)
    dates = makeSchedule(
        start,
        ref_date,
        frequency,
        calendar="china.sse",
        dateRule=BizDayConventions.Following,
        dateGenerationRule=DateGeneration.Backward,
    )
    date_strings = [value.strftime("%Y-%m-%d") for value in dates]
    factor_df = engine.fetch_factor_range(universe, factors=transformer, dates=date_strings)
    target_df = _fit_target(
        engine, universe, date_strings, fit_target, map_freq(frequency), frequency
    )
    return factor_df, target_df, date_strings


def fetch_train_phase(
    engine,
    alpha_factors,
    ref_date,
    frequency,
    universe,
    batch=1,
    neutralized_risk=None,
    risk_model="short",
    pre_process=None,
    post_process=None,
    warm_start=0,
    fit_target=None,
) -> dict:
    transformer = (
        alpha_factors if isinstance(alpha_factors, Transformer) else Transformer(alpha_factors)
    )
    factor_df, target_df, date_strings = _training_window(
        engine, transformer, ref_date, frequency, universe, batch, warm_start, fit_target
    )
    keys = ["trade_date", "code"]
    data = drop_nulls(factor_df.join(target_df, keys=keys, join_type="inner"))
    target = data.select(keys + ["dx"])
    factors = data.select(keys + transformer.names)
    target, dates, date_label, risk_exp, x_values, y_values, _, _, codes = _merge_df(
        engine,
        transformer.names,
        factors,
        target,
        universe,
        date_strings,
        risk_model,
        neutralized_risk,
    )
    ref_timestamp = timestamp_scalar(ref_date).as_py()
    if dates[-1] == ref_timestamp:
        if len(dates) < 2:
            raise ValueError(f"No previous data for training for the date {ref_date}")
        end = dates[-2]
        start = dates[-batch - 1] if batch <= len(dates) - 1 else dates[0]
    else:
        end = dates[-1]
        start = dates[-batch] if batch <= len(dates) else dates[0]
    index = (date_label >= start) & (date_label <= end)
    risks = risk_exp[index] if risk_exp is not None else None
    processed_x = factor_processing(
        x_values[index], pre_process=pre_process, risk_factors=risks, post_process=post_process
    )
    processed_y = factor_processing(
        y_values[index], pre_process=pre_process, risk_factors=risks, post_process=post_process
    )
    return {
        "x_names": transformer.names,
        "train": {
            "x": from_numpy(processed_x, transformer.names),
            "y": processed_y,
            "code": codes[index],
        },
    }


def fetch_predict_phase(
    engine,
    alpha_factors,
    ref_date,
    frequency,
    universe,
    batch=1,
    neutralized_risk=None,
    risk_model="short",
    pre_process=None,
    post_process=None,
    warm_start=0,
    fillna=None,
    fit_target=None,
):
    transformer = (
        alpha_factors if isinstance(alpha_factors, Transformer) else Transformer(alpha_factors)
    )
    factor_df, target_df, date_strings = _training_window(
        engine, transformer, ref_date, frequency, universe, batch, warm_start, fit_target
    )
    factor_df = _median_fill(factor_df, transformer.names) if fillna else drop_nulls(factor_df)
    keys = ["trade_date", "code"]
    if neutralized_risk:
        risk_df = engine.fetch_risk_model_range(
            universe, dates=sorted(set(factor_df["trade_date"].to_pylist())), risk_model=risk_model
        )[1]
        used = sorted(set(neutralized_risk).difference(transformer.names))
        risk_df = drop_nulls(risk_df.select(keys + used))
        train_x = factor_df.join(risk_df, keys=keys, join_type="inner")
        train_x = train_x.join(target_df, keys=keys, join_type="left outer")
        risk_exp = to_numpy(train_x, neutralized_risk, dtype=float)
    else:
        train_x = factor_df.join(target_df, keys=keys, join_type="left outer")
        risk_exp = None
    train_x = drop_nulls(train_x, [name for name in train_x.column_names if name != "dx"])
    x_values = to_numpy(train_x, transformer.names, dtype=float)
    y_values = to_numpy(train_x, ["dx"], dtype=float)
    date_label = np.asarray(train_x["trade_date"].to_pylist())
    dates = np.unique(date_label)
    if not len(dates) or dates[-1] != timestamp_scalar(ref_date).as_py():
        return {
            "x_names": transformer.names,
            "predict": {
                "x": from_numpy(None, transformer.names),
                "code": np.array([]),
                "y": np.array([]),
            },
        }
    end = dates[-1]
    start = dates[-batch] if batch <= len(dates) else dates[0]
    left = bisect.bisect_left(date_label, start)
    right = bisect.bisect_right(date_label, end)
    risks = risk_exp[left:right] if risk_exp is not None else None
    processed_x = factor_processing(
        x_values[left:right], pre_process=pre_process, risk_factors=risks, post_process=post_process
    )
    processed_y = factor_processing(
        y_values[left:right], pre_process=pre_process, risk_factors=risks, post_process=post_process
    )
    sub_dates = date_label[left:right]
    inner_left = bisect.bisect_left(sub_dates, end)
    inner_right = bisect.bisect_right(sub_dates, end)
    processed_x = processed_x[inner_left:inner_right]
    processed_y = processed_y[inner_left:inner_right]
    codes = column_values(train_x, "code")[left:right][inner_left:inner_right]
    return {
        "x_names": transformer.names,
        "predict": {
            "x": from_numpy(processed_x, transformer.names),
            "code": codes,
            "y": processed_y.flatten(),
        },
    }
