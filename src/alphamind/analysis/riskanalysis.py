import numpy as np
import pyarrow as pa

from alphamind.data.arrow import as_arrow, column_values, to_numpy
from alphamind.data.neutralize import neutralize


def risk_analysis(net_weights: pa.Table, next_bar_returns: pa.Table, risk_table: pa.Table):
    net_table = as_arrow(net_weights)
    return_table = as_arrow(next_bar_returns)
    risks = as_arrow(risk_table)
    if not (net_table.num_rows == return_table.num_rows == risks.num_rows):
        raise ValueError("weight, return and risk tables must have the same rows")
    if "trade_date" not in net_table.column_names:
        raise ValueError("net_weights requires a trade_date column")
    dates = np.asarray(net_table["trade_date"].to_pylist())
    _, group_idx = np.unique(dates, return_inverse=True)
    net_pos = column_values(net_table, "weight").reshape((-1, 1))
    returns = column_values(return_table, "return")
    factor_names = [name for name in risks.column_names if name not in ("trade_date", "code")]
    idiosyncratic, stats = neutralize(
        to_numpy(risks, factor_names, dtype=float), returns, group_idx, detail=True
    )
    explained = np.hstack((idiosyncratic, stats["explained"][:, :, 0])) * net_pos
    exposure = stats["exposure"][:, :, 0] * net_pos
    key_data = {
        name: net_table[name] for name in ("trade_date", "code") if name in net_table.column_names
    }
    explained_table = pa.table(
        {
            **key_data,
            **{
                name: explained[:, index]
                for index, name in enumerate(["idiosyncratic"] + factor_names)
            },
        }
    )
    _, first_rows = np.unique(group_idx, return_index=True)
    exposure_table = pa.table(
        {
            "trade_date": net_table["trade_date"].take(pa.array(first_rows)),
            **{name: exposure[first_rows, index] for index, name in enumerate(factor_names)},
        }
    )
    return explained_table, exposure_table
