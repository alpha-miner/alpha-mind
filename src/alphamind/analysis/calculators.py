import numpy as np
import pyarrow as pa

from alphamind.data.arrow import as_arrow


def calculate_turn_over(pos_table: pa.Table) -> pa.Table:
    table = as_arrow(pos_table).sort_by("trade_date")
    dates = table["trade_date"].to_pylist()
    codes = table["code"].to_pylist()
    unique_dates = list(dict.fromkeys(dates))
    factors = [name for name in table.column_names if name not in ("trade_date", "code")]
    result = {"trade_date": unique_dates}
    for name in factors:
        values = table[name].to_pylist()
        previous = {}
        turnovers = []
        for date in unique_dates:
            current = {
                code: value
                for row_date, code, value in zip(dates, codes, values)
                if row_date == date
            }
            turnovers.append(
                float(
                    np.sum(
                        [
                            abs(current.get(code, 0.0) - previous.get(code, 0.0))
                            for code in set(current).union(previous)
                        ]
                    )
                )
            )
            previous = current
        result[name] = turnovers
    return pa.table(result)
