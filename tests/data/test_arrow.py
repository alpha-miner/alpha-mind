import datetime as dt

import numpy as np
import pyarrow as pa
import sqlalchemy as sa

from alphamind.data.arrow import (
    as_arrow,
    drop_nulls,
    read_parquet,
    read_sql,
    read_table,
    replace_inf,
    to_numpy,
    write_parquet,
    write_table,
)


def test_parquet_round_trip_uses_arrow_schema(tmp_path):
    table = pa.table(
        {
            "trade_date": [dt.date(2026, 1, 5), dt.date(2026, 1, 6)],
            "code": ["000001", "000002"],
            "value": [1.5, None],
        }
    )
    path = tmp_path / "factors.parquet"

    write_parquet(table, path)
    restored = read_parquet(path)

    assert isinstance(restored, pa.Table)
    assert restored.equals(as_arrow(table))
    assert read_table(path, columns=["code"]).column_names == ["code"]


def test_ipc_round_trip(tmp_path):
    table = pa.table({"code": [1, 2], "value": [3.0, 4.0]})
    path = tmp_path / "factors.arrow"

    write_table(table, path)

    assert read_table(path).equals(table)


def test_sql_query_returns_arrow_without_pandas(tmp_path):
    engine = sa.create_engine("sqlite+pysqlite:///" + str(tmp_path / "query.sqlite"))
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE factor (trade_date DATE, code TEXT, value FLOAT)"))
        conn.execute(
            sa.text("INSERT INTO factor VALUES (:date, :code, :value)"),
            [
                {"date": dt.date(2026, 1, 5), "code": "000001", "value": 1.0},
                {"date": dt.date(2026, 1, 6), "code": "000002", "value": 2.0},
            ],
        )

    result = read_sql(sa.text("SELECT * FROM factor ORDER BY code"), engine)

    assert isinstance(result, pa.Table)
    assert result["code"].to_pylist() == ["000001", "000002"]
    assert pa.types.is_timestamp(result.schema.field("trade_date").type)


def test_arrow_cleaning_and_numpy_boundary():
    table = pa.table({"x": [1.0, None, np.inf], "y": [2.0, 3.0, 4.0]})

    cleaned = drop_nulls(replace_inf(table))

    assert cleaned.to_pydict() == {"x": [1.0], "y": [2.0]}
    np.testing.assert_array_equal(to_numpy(cleaned), np.array([[1.0, 2.0]]))


def test_connectorx_binary_text_is_normalized_for_calc_flow():
    table = pa.table(
        {
            "name": pa.array([b"alpha", b"beta"], type=pa.large_binary()),
        }
    )

    result = as_arrow(table)

    assert result.schema.field("name").type == pa.string()
    assert result["name"].to_pylist() == ["alpha", "beta"]


def test_record_batch_reader_converts_all_batches():
    table = pa.table(
        {
            "code": ["000001", "000002", "000003"],
            "price": [100.0, 200.0, 300.0],
        }
    )

    batches = table.to_batches(max_chunksize=1)
    reader = pa.RecordBatchReader.from_batches(table.schema, batches)

    result = as_arrow(reader)

    assert isinstance(result, pa.Table)
    assert result.equals(table)
    assert result.num_rows == 3
