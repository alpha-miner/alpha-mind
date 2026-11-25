import pyarrow as pa
import pyarrow.compute as pc

from alphamind.data.arrow import as_arrow


def _rectangular_width(table, dates):
    if len(dates) < 2 or table.num_rows % len(dates):
        return None
    width = table.num_rows // len(dates)
    counts = pc.value_counts(table["trade_date"]).field("counts")
    if not pc.all(pc.equal(counts, width)).as_py():
        return None
    codes = table["code"].combine_chunks()
    first_codes = codes.slice(0, width)
    if len(pc.unique(first_codes)) != width:
        return None
    repeated = pa.concat_arrays([first_codes] * (len(dates) - 1))
    return width if codes.slice(width).equals(repeated) else None


def _rectangular_turnover(table, dates, factors, width):
    periods = pc.cast(pc.index_in(table["trade_date"], value_set=dates), pa.int64())
    changes = []
    first_values = []
    change_names = []
    for index, name in enumerate(factors):
        values = pc.cast(table[name], pa.float64()).combine_chunks()
        first = pc.sum(pc.abs(values.slice(0, width))).as_py()
        changes.append(
            pc.abs(pc.subtract(values.slice(width), values.slice(0, len(values) - width)))
        )
        first_values.append(first)
        change_names.append(f"__change_{index}")

    grouped = pa.Table.from_arrays(
        [periods.slice(width)] + changes, names=["__period"] + change_names
    )
    result = grouped.group_by("__period").aggregate([(name, "sum") for name in change_names])
    result = result.take(pc.sort_indices(result["__period"]))
    return pa.Table.from_arrays(
        [dates]
        + [
            pa.concat_arrays(
                [pa.array([first], type=pa.float64()), result[f"{name}_sum"].combine_chunks()]
            )
            for first, name in zip(first_values, change_names)
        ],
        names=["trade_date"] + factors,
    )


def _aligned_turnover(table, dates, factors):
    prefix = "__alphamind_turnover"
    while any(name.startswith(prefix) for name in table.column_names):
        prefix = "_" + prefix
    period_name = f"{prefix}_period"
    periods = pc.cast(pc.index_in(table["trade_date"], value_set=dates), pa.int64())
    keyed = pa.Table.from_arrays(
        [periods, table["code"].combine_chunks()]
        + [pc.cast(table[name], pa.float64()).combine_chunks() for name in factors],
        names=[period_name, "code"] + factors,
    )
    deduped = keyed.group_by([period_name, "code"], use_threads=False).aggregate(
        [(name, "last") for name in factors]
    )

    periods = deduped[period_name].combine_chunks()
    codes = deduped["code"].combine_chunks()
    next_periods = pc.add(periods, 1)
    has_next = pc.less(next_periods, len(dates))
    previous_periods = next_periods.filter(has_next)
    previous_codes = codes.filter(has_next)
    count = deduped.num_rows
    previous_count = len(previous_periods)
    paired_columns = {
        period_name: pa.concat_arrays([periods, previous_periods]),
        "code": pa.concat_arrays([codes, previous_codes]),
    }
    for name in factors:
        values = deduped[f"{name}_last"].combine_chunks()
        paired_columns[f"{prefix}_current_{name}"] = pa.concat_arrays(
            [values, pa.nulls(previous_count, pa.float64())]
        )
        paired_columns[f"{prefix}_previous_{name}"] = pa.concat_arrays(
            [pa.nulls(count, pa.float64()), values.filter(has_next)]
        )

    paired = pa.table(paired_columns)
    aggregations = [
        (f"{prefix}_{side}_{name}", "sum") for name in factors for side in ("current", "previous")
    ]
    aligned = paired.group_by([period_name, "code"]).aggregate(aggregations)
    changes = []
    change_names = []
    for name in factors:
        current = pc.fill_null(aligned[f"{prefix}_current_{name}_sum"], 0.0)
        previous = pc.fill_null(aligned[f"{prefix}_previous_{name}_sum"], 0.0)
        changes.append(pc.abs(pc.subtract(current, previous)))
        change_names.append(f"{prefix}_change_{name}")

    aligned = pa.Table.from_arrays(
        [aligned[period_name].combine_chunks()] + changes,
        names=[period_name] + change_names,
    )
    result = aligned.group_by(period_name).aggregate([(name, "sum") for name in change_names])
    result = result.take(pc.sort_indices(result[period_name]))
    return pa.Table.from_arrays(
        [pc.take(dates, result[period_name])] + [result[f"{name}_sum"] for name in change_names],
        names=["trade_date"] + factors,
    )


def calculate_turn_over(pos_table: pa.Table) -> pa.Table:
    table = as_arrow(pos_table).sort_by("trade_date")
    factors = [name for name in table.column_names if name not in ("trade_date", "code")]
    dates = pc.unique(table["trade_date"])
    if table.num_rows == 0:
        return pa.Table.from_arrays(
            [dates] + [pa.array([], type=pa.float64()) for _ in factors],
            names=["trade_date"] + factors,
        )
    width = _rectangular_width(table, dates)
    if width is not None:
        return _rectangular_turnover(table, dates, factors, width)
    return _aligned_turnover(table, dates, factors)
