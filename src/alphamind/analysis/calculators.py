import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from alphamind.data.arrow import as_arrow


def _date_slices(table):
    dates = table["trade_date"].combine_chunks()
    if pa.types.is_timestamp(dates.type) or pa.types.is_date(dates.type):
        values = pc.cast(dates, pa.int64()).to_numpy(zero_copy_only=False)
    else:
        values = np.asarray(dates.to_pylist())
    if not len(values):
        empty = np.array([], dtype=np.int64)
        return dates, empty, empty
    starts = np.concatenate(([0], np.flatnonzero(values[1:] != values[:-1]) + 1)).astype(
        np.int64, copy=False
    )
    ends = np.concatenate((starts[1:], [len(values)])).astype(np.int64, copy=False)
    return dates, starts, ends


def calculate_turn_over(pos_table: pa.Table) -> pa.Table:
    table = as_arrow(pos_table).sort_by("trade_date")
    dates, starts, ends = _date_slices(table)
    factors = [name for name in table.column_names if name not in ("trade_date", "code")]
    result = {"trade_date": dates.take(pa.array(starts))}
    if table.num_rows == 0:
        for name in factors:
            result[name] = pa.array([], type=pa.float64())
        return pa.table(result)

    codes = table["code"].to_pylist()
    first_codes = codes[: ends[0]]
    is_rectangular = len(set(first_codes)) == len(first_codes) and np.all(
        ends - starts == len(first_codes)
    )
    if is_rectangular:
        is_rectangular = all(
            codes[start:end] == first_codes for start, end in zip(starts[1:], ends[1:])
        )

    for name in factors:
        if is_rectangular:
            values = table[name].to_numpy(zero_copy_only=False).astype(float, copy=False)
            matrix = values.reshape(len(starts), len(first_codes))
            result[name] = np.abs(
                np.diff(
                    matrix,
                    axis=0,
                    prepend=np.zeros((1, len(first_codes)), dtype=float),
                )
            ).sum(axis=1)
            continue

        values = table[name].to_pylist()
        previous = {}
        turnovers = []
        for start, end in zip(starts, ends):
            current = dict(zip(codes[start:end], values[start:end]))
            changed_codes = set(current).union(previous)
            turnovers.append(
                float(
                    np.sum(
                        [
                            abs(current.get(code, 0.0) - previous.get(code, 0.0))
                            for code in changed_codes
                        ]
                    )
                )
            )
            previous = current
        result[name] = turnovers
    return pa.table(result)
