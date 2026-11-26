import datetime as dt
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import sqlalchemy as sa

from alphamind.data.arrow import (
    align_values,
    as_arrow,
    as_arrow_array,
    column_values,
    drop_duplicates,
    drop_nulls,
    empty_table,
    from_numpy,
    last_index_in,
    read_sql,
    read_table,
    timestamp_scalar,
    to_numpy,
    with_column,
    write_table,
)


def test_as_arrow_supports_batches_readers_and_to_pyarrow():
    table = pa.table({"value": [1.0, 2.0]})
    batch = table.to_batches()[0]

    assert as_arrow(batch).equals(table)
    assert as_arrow(pa.RecordBatchReader.from_batches(table.schema, [batch])).equals(table)

    class Convertible:
        def to_pyarrow(self):
            return table

    assert as_arrow(Convertible()).equals(table)

    schema = pa.schema([pa.field("renamed", pa.float64())])

    assert as_arrow(table, schema=schema).column_names == ["renamed"]
    assert as_arrow(batch, schema=schema).column_names == ["renamed"]

    reader = pa.RecordBatchReader.from_batches(table.schema, [batch])
    assert as_arrow(reader, schema=schema).column_names == ["renamed"]

    assert as_arrow(Convertible(), schema=schema).column_names == ["renamed"]

    with pytest.raises(TypeError, match="Expected Arrow tabular data"):
        as_arrow([1, 2])


def test_arrow_normalizes_temporal_text_decimal_and_binary_types():
    table = pa.table(
        {
            "trade_date": ["2026-01-05", "2026-01-06"],
            "updatetime": [None, None],
            "name": pa.array(["A", "B"], type=pa.large_string()),
            "payload": pa.array([b"a", b"b"], type=pa.binary()),
            "amount": pa.array([Decimal("1.25"), Decimal("2.50")]),
        }
    )

    result = as_arrow(table)

    assert result.schema.field("trade_date").type == pa.timestamp("us", tz="UTC")
    assert result.schema.field("updatetime").type == pa.timestamp("us", tz="UTC")
    assert result.schema.field("name").type == pa.string()
    assert result.schema.field("payload").type == pa.string()
    assert result.schema.field("amount").type == pa.float64()
    assert result["amount"].to_pylist() == [1.25, 2.5]


def test_numpy_conversion_and_column_mutation():
    assert empty_table(["x", "y"]).schema.names == ["x", "y"]
    assert from_numpy(None, ["x"]).num_rows == 0
    assert from_numpy(np.array([1.0, 2.0]), ["x"]).shape == (2, 1)

    with pytest.raises(ValueError, match="one column per name"):
        from_numpy(np.ones((2, 2)), ["x"])

    table = pa.table({"x": [1, 2]})
    appended = with_column(table, "y", [3, 4])
    replaced = with_column(appended, "x", pa.array([5, 6]))

    assert appended.column_names == ["x", "y"]
    assert replaced["x"].to_pylist() == [5, 6]
    np.testing.assert_array_equal(column_values(replaced, "x"), [5, 6])
    assert to_numpy(replaced, columns=[], dtype=float).shape == (2, 0)
    assert to_numpy(replaced, ["x"], dtype=float).dtype == np.dtype(float)


def test_null_duplicate_and_timestamp_helpers():
    table = pa.table({"x": [1.0, np.nan, None, 1.0], "y": [1, 2, 3, 1]})

    assert drop_nulls(table, [], drop_nan=True).num_rows == 4
    assert drop_nulls(table, ["y"]).num_rows == 4
    assert drop_nulls(table, ["x"], drop_nan=False).num_rows == 3
    assert drop_duplicates(table, ["x", "y"]).num_rows == 3

    expected = pa.scalar(
        dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc),
        pa.timestamp("us", "UTC"),
    )
    assert timestamp_scalar("2026-01-05") == expected
    assert timestamp_scalar(dt.date(2026, 1, 5)) == expected
    assert timestamp_scalar(dt.datetime(2026, 1, 5)) == expected

    aware = dt.datetime(2026, 1, 5, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    assert timestamp_scalar(aware).as_py().utcoffset() == dt.timedelta(0)


def test_arrow_alignment_preserves_last_values_and_missing_defaults():
    source_codes = pa.chunked_array([["A", "A"], ["B"]])
    values = pa.chunked_array([[0.1], [0.2, 0.3]])
    codes = pa.array(["B", "A", "C"])

    aligned = align_values(codes, source_codes, values)

    assert aligned.to_pylist() == pytest.approx([0.3, 0.2, 0.0])
    assert last_index_in(codes, source_codes).to_pylist() == [2, 1, None]
    assert align_values([2, 1], [1, 2], [10.0, 20.0]).to_pylist() == [20.0, 10.0]
    assert np.isnan(align_values(["C"], ["A"], [1.0], default=np.nan)[0].as_py())
    assert last_index_in(pa.array([None]), pa.array(["A", None])).to_pylist() == [1]
    assert last_index_in(pa.array(["A"]), pa.array([None])).to_pylist() == [None]
    assert last_index_in([None], [None]).to_pylist() == [0]
    assert last_index_in([], []).to_pylist() == []
    assert align_values(["A"], [], []).to_pylist() == [0.0]
    assert as_arrow_array(source_codes).equals(pa.array(["A", "A", "B"]))


def test_drop_duplicates_uses_arrow_groups_and_preserves_first_row_order():
    table = pa.table(
        {
            "code": pa.chunked_array([["B", "A"], ["B", "A", "A"]]),
            "group": [1, 1, 1, 1, 2],
            "value": ["first-b", "first-a", "duplicate-b", "duplicate-a", "first-a2"],
            "__alphamind_row_id": [10, 11, 12, 13, 14],
        }
    )

    result = drop_duplicates(table, ["code", "group"])

    assert result["value"].to_pylist() == ["first-b", "first-a", "first-a2"]
    assert result.column_names == table.column_names
    assert drop_duplicates(table, []).num_rows == 0
    assert drop_duplicates(table.slice(0, 0), ["code"]).schema == table.schema


def test_csv_feather_parquet_and_stream_round_trips(tmp_path):
    table = pa.table({"code": ["A", "B"], "value": [1.0, 2.0]})

    for suffix in ("csv", "feather", "pq"):
        path = tmp_path / f"nested/data.{suffix}"
        assert write_table(table, path) == path
        restored = read_table(path)
        assert restored["code"].to_pylist() == ["A", "B"]
        assert restored["value"].to_pylist() == [1.0, 2.0]

    parquet = tmp_path / "data.parquet"
    write_table(table, parquet)

    filtered = read_table(parquet, columns=["code"], filters=[("value", ">", 1.0)])
    assert filtered.to_pydict() == {"code": ["B"]}
    assert read_table(parquet, file_format="PARQUET").column_names == ["code", "value"]

    stream = tmp_path / "stream.arrow"

    with pa.OSFile(str(stream), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    assert read_table(stream).equals(table)


def test_file_format_validation(tmp_path):
    table = pa.table({"x": [1]})
    ipc_path = tmp_path / "data.arrow"
    csv_path = tmp_path / "data.csv"

    write_table(table, ipc_path)
    write_table(table, csv_path)

    with pytest.raises(ValueError, match="only supported for parquet"):
        read_table(ipc_path, columns=["x"])

    with pytest.raises(ValueError, match="filters are only supported"):
        read_table(csv_path, filters=[("x", ">", 0)])

    with pytest.raises(ValueError, match="Cannot infer file format"):
        read_table(tmp_path / "data.unknown")

    with pytest.raises(ValueError, match="Unsupported file format"):
        read_table(tmp_path / "data.unknown", file_format="json")

    with pytest.raises(ValueError, match="Unsupported file format"):
        write_table(table, tmp_path / "data.unknown", file_format="json")


def test_read_sql_parameter_and_connection_variants(tmp_path):
    url = "sqlite+pysqlite:///" + str(tmp_path / "parameters.sqlite")
    engine = sa.create_engine(url)

    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE values_table (value DOUBLE, name TEXT)"))
        connection.execute(sa.text("INSERT INTO values_table VALUES (1.0, 'A'), (2.0, 'B')"))
        connection.execute(sa.text("SELECT * FROM values_table"))

    with engine.connect() as connection:
        result = read_sql(
            "SELECT name FROM values_table WHERE value > :minimum ORDER BY name",
            connection,
            {"minimum": 1.0},
        )

    assert result["name"].to_pylist() == ["B"]

    assert read_sql("SELECT value FROM values_table ORDER BY value", url)["value"].to_pylist() == [
        1.0,
        2.0,
    ]

    with pytest.raises(TypeError, match="must be a mapping"):
        read_sql("SELECT * FROM values_table WHERE value > :minimum", engine, [1.0])
