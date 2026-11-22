import calc_flow as cf
import numpy as np
import pyarrow as pa


def simple_settle(
    weights: np.ndarray,
    dx_return: np.ndarray,
    groups: np.ndarray = None,
    benchmark: np.ndarray = None,
) -> pa.Table:
    weights = np.asarray(weights).flatten()
    returns = np.asarray(dx_return).flatten()
    net_pos = weights - np.asarray(benchmark).flatten() if benchmark is not None else weights
    labels = (
        ["total"] * len(weights) if groups is None else [str(value) for value in np.asarray(groups)]
    )
    table = pa.table(
        {
            "industry": labels,
            "weight": net_pos,
            "returns": returns,
            "contribution": net_pos * returns,
        }
    )
    total_query = """
        SELECT 'total' AS industry, SUM(contribution) AS er, CORR(weight, returns) AS ic
        FROM input
    """
    total = cf.compute(table, lambda input_table: input_table.sql(total_query))
    if groups is None:
        return total
    grouped_query = """
        SELECT industry, SUM(contribution) AS er, CORR(weight, returns) AS ic
        FROM input
        GROUP BY industry
        ORDER BY industry
    """
    grouped = cf.compute(table, lambda input_table: input_table.sql(grouped_query))
    grouped_labels = grouped["industry"].to_pylist()
    expected_order = [str(value) for value in np.unique(groups)]
    grouped = grouped.take(pa.array([grouped_labels.index(label) for label in expected_order]))
    schema = pa.schema([pa.field(field.name, field.type) for field in grouped.schema])
    grouped = pa.Table.from_arrays(grouped.columns, schema=schema)
    total = pa.Table.from_arrays(total.columns, schema=schema)
    return pa.concat_tables([grouped, total])
