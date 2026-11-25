import calc_flow as cf
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from alphamind.data.arrow import as_arrow, as_arrow_array, last_index_in


def simple_settle(
    weights: np.ndarray,
    dx_return: np.ndarray,
    groups: np.ndarray = None,
    benchmark: np.ndarray = None,
) -> pa.Table:
    weights = np.asarray(weights).flatten()
    returns = np.asarray(dx_return).flatten()
    net_pos = weights - np.asarray(benchmark).flatten() if benchmark is not None else weights
    group_values = None if groups is None else as_arrow_array(np.asarray(groups))
    labels = (
        pa.repeat("total", len(weights))
        if group_values is None
        else pc.fill_null(pc.cast(group_values, pa.string()), "None")
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
    combined = as_arrow(cf.compute(table, lambda input_table: input_table.sql(combined_query)))
    unique_groups = pc.unique(group_values)
    sorted_groups = pc.take(unique_groups, pc.sort_indices(unique_groups))
    expected_groups = pc.fill_null(pc.cast(sorted_groups, pa.string()), "None")
    expected_order = pa.concat_arrays([expected_groups, pa.array(["total"])])
    combined = combined.take(last_index_in(expected_order, combined["industry"]))
    schema = pa.schema([pa.field(field.name, field.type) for field in combined.schema])
    return pa.Table.from_arrays(combined.columns, schema=schema)
