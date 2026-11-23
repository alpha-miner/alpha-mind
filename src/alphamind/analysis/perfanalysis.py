import pyarrow as pa

from alphamind.analysis.riskanalysis import risk_analysis


def perf_attribution_by_pos(
    net_weights: pa.Table, next_bar_returns: pa.Table, benchmark_table: pa.Table
) -> pa.Table:
    explained, _ = risk_analysis(net_weights, next_bar_returns, benchmark_table)
    value_columns = [name for name in explained.column_names if name not in ("trade_date", "code")]
    grouped = explained.group_by("trade_date").aggregate([(name, "sum") for name in value_columns])
    return grouped.rename_columns(["trade_date"] + value_columns).sort_by("trade_date")
