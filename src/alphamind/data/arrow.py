from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import connectorx as cx
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import sqlalchemy as sa

ArrowTable = pa.Table
TableInput = Union[pa.Table, pa.RecordBatch, pa.RecordBatchReader]


def _normalize_temporal(table: pa.Table) -> pa.Table:
    for index, field in enumerate(table.schema):
        temporal_name = field.name.lower().endswith("date") or field.name.lower() == "updatetime"
        if pa.types.is_date(field.type) or (
            temporal_name
            and (
                pa.types.is_string(field.type)
                or pa.types.is_null(field.type)
                or pa.types.is_timestamp(field.type)
            )
        ):
            column = table.column(index)
            if not pa.types.is_timestamp(column.type):
                column = pc.cast(column, pa.timestamp("us"))
            if column.type.tz is None:
                column = pc.assume_timezone(column, "UTC")
            elif column.type != pa.timestamp("us", tz="UTC"):
                column = pc.cast(column, pa.timestamp("us", tz="UTC"))
            table = table.set_column(index, field.name, column)
        elif (
            pa.types.is_binary(field.type)
            or pa.types.is_large_binary(field.type)
            or pa.types.is_large_string(field.type)
        ):
            table = table.set_column(
                index,
                pa.field(field.name, pa.string(), nullable=field.nullable),
                pc.cast(table.column(index), pa.string()),
            )
        elif pa.types.is_decimal(field.type):
            table = table.set_column(
                index,
                pa.field(field.name, pa.float64(), nullable=field.nullable),
                pc.cast(table.column(index), pa.float64()),
            )
    return table


def as_arrow(data: TableInput, schema: pa.Schema | None = None) -> pa.Table:
    if isinstance(data, pa.Table):
        table = data if schema is None else pa.Table.from_arrays(data.columns, schema=schema)
    elif isinstance(data, pa.RecordBatch):
        table = pa.Table.from_batches([data], schema=schema)
    elif isinstance(data, pa.RecordBatchReader):
        table = data.read_all()
    elif hasattr(data, "to_pyarrow"):
        table = data.to_pyarrow()
    else:
        raise TypeError(f"Expected Arrow tabular data, got {type(data).__name__}")
    return _normalize_temporal(table)


def from_numpy(values, names: Sequence[str]) -> pa.Table:
    if values is None:
        return empty_table(names)
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1, 1))
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("values must be a matrix with one column per name")
    return pa.Table.from_arrays(
        [pa.array(matrix[:, index]) for index in range(matrix.shape[1])], names=list(names)
    )


def empty_table(names: Iterable[str] = ()) -> pa.Table:
    return pa.Table.from_arrays([pa.array([], type=pa.float64()) for _ in names], names=list(names))


def with_column(table: pa.Table, name: str, values) -> pa.Table:
    table = as_arrow(table)
    array = values if isinstance(values, (pa.Array, pa.ChunkedArray)) else pa.array(values)
    index = table.schema.get_field_index(name)
    if index < 0:
        return table.append_column(name, array)
    return table.set_column(index, name, array)


def column_values(table: pa.Table, name: str) -> np.ndarray:
    return table[name].combine_chunks().to_numpy(zero_copy_only=False)


def to_numpy(table: pa.Table, columns: Sequence[str] | None = None, dtype=None) -> np.ndarray:
    table = as_arrow(table)
    selected = table.select(list(columns)) if columns is not None else table
    if selected.num_columns == 0:
        return np.empty((selected.num_rows, 0), dtype=dtype)
    arrays = [column_values(selected, name) for name in selected.column_names]
    matrix = np.column_stack(arrays)
    return matrix.astype(dtype, copy=False) if dtype is not None else matrix


def drop_nulls(
    table: pa.Table, columns: Sequence[str] | None = None, drop_nan: bool = True
) -> pa.Table:
    names = list(columns) if columns is not None else table.column_names
    if not names:
        return table
    mask = pa.array(np.ones(table.num_rows, dtype=bool))
    for name in names:
        column = table[name]
        valid = pc.is_valid(column)
        if drop_nan and pa.types.is_floating(column.type):
            valid = pc.and_kleene(valid, pc.invert(pc.is_nan(column)))
        mask = pc.and_kleene(mask, valid)
    return table.filter(mask)


def drop_duplicates(table: pa.Table, keys: Sequence[str]) -> pa.Table:
    table = as_arrow(table)
    columns = [table[name].to_pylist() for name in keys]
    seen = set()
    indices = []
    for index, key in enumerate(zip(*columns)):
        if key not in seen:
            seen.add(key)
            indices.append(index)
    return table.take(pa.array(indices, type=pa.int64()))


def timestamp_scalar(value) -> pa.Scalar:
    import datetime as dt

    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value)
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        value = dt.datetime.combine(value, dt.time())
    if isinstance(value, dt.datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return pa.scalar(value, type=pa.timestamp("us", tz="UTC"))


def replace_inf(table: pa.Table) -> pa.Table:
    for index, field in enumerate(table.schema):
        if pa.types.is_floating(field.type):
            column = table.column(index).combine_chunks()
            table = table.set_column(
                index,
                field,
                pc.replace_with_mask(column, pc.is_inf(column), pa.scalar(None, field.type)),
            )
    return table


def read_sql(query, connection, parameters=None) -> pa.Table:
    if hasattr(connection, "bind") and not hasattr(connection, "dialect"):
        connection = connection.bind
    if isinstance(connection, str):
        url = sa.engine.make_url(connection)
        dialect = url.get_dialect()()
    elif isinstance(connection, sa.engine.Connection):
        url = connection.engine.url
        dialect = connection.dialect
    else:
        url = connection.url
        dialect = connection.dialect
    if parameters:
        if not isinstance(parameters, Mapping):
            raise TypeError("ConnectorX parameters must be a mapping")
        if isinstance(query, str):
            query = sa.text(query)
        query = query.bindparams(
            *[sa.bindparam(name, value=value) for name, value in parameters.items()]
        )
    if not isinstance(query, str):
        query = str(query.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    connector_url = url.set(drivername=url.drivername.split("+", 1)[0])
    result = cx.read_sql(
        connector_url.render_as_string(hide_password=False), query, return_type="arrow"
    )
    return as_arrow(result)


def read_table(
    path, file_format: str | None = None, columns=None, filters=None, **options
) -> pa.Table:
    source = Path(path)
    file_format = _file_format(source, file_format)
    if file_format == "parquet":
        table = pq.read_table(source, columns=columns, filters=filters, **options)
    elif file_format in ("ipc", "feather"):
        if columns is not None or filters is not None:
            raise ValueError("columns and filters are only supported for parquet")
        table = (
            feather.read_table(source, **options) if file_format == "feather" else _read_ipc(source)
        )
    elif file_format == "csv":
        if filters is not None:
            raise ValueError("filters are only supported for parquet")
        table = csv.read_csv(source, **options)
        if columns is not None:
            table = table.select(list(columns))
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    return _normalize_temporal(table)


def write_table(table: TableInput, path, file_format: str | None = None, **options) -> Path:
    target = Path(path)
    arrow_table = as_arrow(table)
    file_format = _file_format(target, file_format)
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "parquet":
        options.setdefault("compression", "zstd")
        pq.write_table(arrow_table, target, **options)
    elif file_format in ("ipc", "feather"):
        feather.write_feather(arrow_table, target, **options)
    elif file_format == "csv":
        csv.write_csv(arrow_table, target, **options)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    return target


def read_parquet(path, **options) -> pa.Table:
    return read_table(path, "parquet", **options)


def write_parquet(table: TableInput, path, **options) -> Path:
    return write_table(table, path, "parquet", **options)


def _file_format(path: Path, file_format: str | None) -> str:
    if file_format is not None:
        return file_format.lower()
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return "parquet"
    if suffix in (".arrow", ".ipc"):
        return "ipc"
    if suffix == ".feather":
        return "feather"
    if suffix == ".csv":
        return "csv"
    raise ValueError("Cannot infer file format; pass file_format explicitly")


def _read_ipc(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_file(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_stream(source).read_all()
