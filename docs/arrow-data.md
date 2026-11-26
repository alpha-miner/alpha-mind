# Arrow and Files

## Data boundaries

`alphamind.data.arrow` is the boundary between files, databases, and internal numerical calculations. Public read and write functions return `pyarrow.Table`; `pa.RecordBatch` and `pa.RecordBatchReader` are also accepted as write inputs.

```python
import pyarrow as pa
from alphamind.data import read_table, write_table

source = pa.table({"code": ["000001", "000002"], "value": [1.0, 2.0]})
write_table(source, "factor.parquet")
loaded = read_table("factor.parquet")
assert isinstance(loaded, pa.Table)
```

Do not use `.to_pandas()` to exchange data between modules. Convert explicitly at a numerical function boundary when matrix operations are required:

```python
from alphamind.data.arrow import column_values, to_numpy

matrix = to_numpy(loaded, ["value"], dtype=float)  # shape: (n, 1)
vector = column_values(loaded, "value")  # shape: (n,)
```

## Supported formats

| Format | Extensions | Read | Write | Notes |
| --- | --- | --- | --- | --- |
| Parquet | `.parquet`, `.pq` | `read_parquet` / `read_table` | `write_parquet` / `write_table` | Zstandard compression by default |
| Arrow IPC | `.arrow`, `.ipc` | `read_table` | `write_table` | Reads IPC files or streams |
| Feather | `.feather` | `read_table` | `write_table` | Arrow Feather files |
| CSV | `.csv` | `read_table` | `write_table` | Accepts PyArrow CSV options |

When the extension is insufficient, pass `file_format="parquet"` explicitly. An unrecognized format raises `ValueError`.

## Parquet column pruning and filtering

```python
from alphamind.data import read_parquet, write_parquet

write_parquet(source, "factor.parquet", compression="zstd", compression_level=5)
selected = read_parquet(
    "factor.parquet",
    columns=["code", "value"],
    filters=[("code", "in", ["000001", "000002"])],
)
```

`columns` reduces the number of columns read. `filters` uses PyArrow filter expressions and can be pushed down into the read. Other keyword arguments are passed to `pyarrow.parquet.read_table` or `write_table`.

IPC and Feather do not support these `columns` / `filters` arguments. CSV supports column selection only after reading and does not support `filters`.

## Type normalization

`as_arrow` and file reads normalize common external types:

- Date, timestamp, and time-series columns whose names end in `date` or equal `updatetime` are converted to `timestamp[us, UTC]`.
- Timezone-naive values are interpreted as UTC; values with other timezones are converted to UTC.
- Binary, large binary, and large string are converted to string.
- Decimal is converted to float64.

Consequently, temporal column types may differ before and after a write/read round trip. Security identifiers should be explicit strings to avoid losing leading zeros; float64 is recommended for risk variances, returns, and weights.

## Common Arrow helpers

```python
from alphamind.data.arrow import (
    align_values,
    as_arrow,
    drop_duplicates,
    drop_nulls,
    from_numpy,
    replace_inf,
    with_column,
)

clean = replace_inf(loaded)
clean = drop_nulls(clean, ["code", "value"])
clean = drop_duplicates(clean, ["code"])
clean = with_column(clean, "value", [2.0, 3.0])
new_table = from_numpy([[1.0, 2.0], [3.0, 4.0]], names=["f1", "f2"])
weights = align_values(["A", "B"], clean["code"], clean["value"], default=0.0)
```

- `drop_nulls` also removes floating-point NaN values by default.
- `drop_duplicates` keeps the first row for each key and does not sort automatically.
- `from_numpy` accepts one- or two-dimensional arrays; a one-dimensional array becomes a single column.
- `with_column` replaces an identically named column or appends one when absent.
- `align_values` aligns codes and values with an Arrow index kernel rather than constructing a Python dictionary row by row. The last value is used for duplicate source codes, and missing target codes receive `default`.

## Data-quality recommendations

Before calculating windowed factors or joining risk data:

1. Fix the schemas of `trade_date`, `code`, and risk columns.
2. Sort by `trade_date` and `code`, then check key uniqueness.
3. Distinguish `null`, NaN, and infinity; do not fill missing windows with zero by default.
4. Ensure asset ordering within each cross-section matches the NumPy risk matrices.
5. Save each research stage's inputs and outputs as Parquet for auditability and reproducibility.

See the [Parquet demo](setup.md) for a complete file workflow.
