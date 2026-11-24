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
    if groups is None:
        total_query = """
            SELECT 'total' AS industry, SUM(contribution) AS er, CORR(weight, returns) AS ic
            FROM input
        """
        return cf.compute(table, lambda input_table: input_table.sql(total_query))
    combined_query = """
        SELECT industry, SUM(contribution) AS er, CORR(weight, returns) AS ic
        FROM input
        GROUP BY industry
        UNION ALL
        SELECT 'total' AS industry, SUM(contribution) AS er, CORR(weight, returns) AS ic
        FROM input
    """
    combined = cf.compute(table, lambda input_table: input_table.sql(combined_query))
    combined_labels = combined["industry"].to_pylist()
    expected_order = [str(value) for value in np.unique(groups)] + ["total"]
    combined = combined.take(pa.array([combined_labels.index(label) for label in expected_order]))
    schema = pa.schema([pa.field(field.name, field.type) for field in combined.schema])
    return pa.Table.from_arrays(combined.columns, schema=schema)
