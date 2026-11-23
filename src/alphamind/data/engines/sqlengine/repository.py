from __future__ import annotations

import datetime as dt

import pyarrow as pa
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from alphamind.data.arrow import as_arrow, read_sql


class ArrowSQLRepository:
    def __init__(self, db_url: str = None, engine=None):
        if engine is None and db_url is None:
            raise ValueError("db_url or engine is required")
        self.engine = engine or sa.create_engine(db_url)

    def create_all(self, model):
        model.metadata.create_all(self.engine)

    def read(self, model, columns=None) -> pa.Table:
        table = self._table(model)
        aliases = self._aliases(model)
        selected = list(table.columns) if columns is None else [table.c[name] for name in columns]
        query = sa.select(
            *[
                column.label(aliases.get(column.key, column.key))
                if column.name != aliases.get(column.key, column.key)
                else column
                for column in selected
            ]
        )
        return read_sql(query, self.engine)

    def write(self, model, data, mode: str = "append", batch_size: int = 1000) -> int:
        table = self._table(model)
        arrow_table = as_arrow(data)
        if arrow_table.num_rows == 0:
            return 0
        records = self._records(table, arrow_table, self._aliases(model))
        if mode not in ("append", "upsert", "replace"):
            raise ValueError("mode must be append, upsert or replace")
        with self.engine.begin() as conn:
            if mode == "replace":
                conn.execute(table.delete())
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                statement = self._upsert(table, batch) if mode == "upsert" else table.insert()
                conn.execute(statement, batch)
        return len(records)

    def _table(self, model):
        if isinstance(model, sa.Table):
            return model
        if isinstance(model, str):
            return sa.Table(model, sa.MetaData(), autoload_with=self.engine)
        return model.__table__

    @staticmethod
    def _aliases(model):
        if not hasattr(model, "__mapper__"):
            return {}
        return {prop.columns[0].key: prop.key for prop in model.__mapper__.column_attrs}

    def _records(self, table, data, aliases):
        source = data.to_pylist()
        records = []
        for row in source:
            record = {}
            for column in table.columns:
                public_name = aliases.get(column.key, column.key)
                if public_name in row:
                    value = row[public_name]
                elif column.key in row:
                    value = row[column.key]
                elif column.name in row:
                    value = row[column.name]
                else:
                    continue
                if (
                    value is not None
                    and isinstance(column.type, sa.Date)
                    and isinstance(value, dt.datetime)
                ):
                    value = value.date()
                elif (
                    value is not None
                    and isinstance(column.type, sa.DateTime)
                    and isinstance(value, dt.datetime)
                    and value.tzinfo is not None
                ):
                    value = value.replace(tzinfo=None)
                record[column.key] = value
            records.append(record)
        return records

    def _upsert(self, table, records):
        dialect = self.engine.dialect.name
        updates = {
            column.key: column
            for column in table.columns
            if not column.primary_key and column.key in records[0]
        }
        if dialect == "mysql":
            statement = mysql_insert(table).values(records)
            values = {name: statement.inserted[column.key] for name, column in updates.items()}
            return statement.on_duplicate_key_update(**values)
        if dialect in ("sqlite", "postgresql"):
            insert = sqlite_insert if dialect == "sqlite" else postgresql_insert
            statement = insert(table).values(records)
            conflict_columns = self._conflict_columns(table, records[0])
            values = {name: statement.excluded[column.key] for name, column in updates.items()}
            return statement.on_conflict_do_update(index_elements=conflict_columns, set_=values)
        raise ValueError(f"Upsert is not supported for dialect {dialect}")

    @staticmethod
    def _conflict_columns(table, record):
        primary = [column.key for column in table.primary_key.columns if column.key in record]
        if primary:
            return primary
        for index in table.indexes:
            if index.unique:
                columns = [column.key for column in index.columns]
                if all(column in record for column in columns):
                    return columns
        raise ValueError("Upsert requires primary-key values or a matching unique index")
