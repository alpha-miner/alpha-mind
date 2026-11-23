# -*- coding: utf-8 -*-
"""
Created on 2017-8-23

@author: cheng.li
"""

import calc_flow as cf
import pyarrow as pa

from alphamind.data.arrow import as_arrow, drop_nulls, empty_table
from alphamind.formula.expressions import Expression


def factor_translator(factor_pool):
    if not factor_pool:
        return None, None
    if isinstance(factor_pool, str):
        return {factor_pool: factor_pool}, [factor_pool]
    if isinstance(factor_pool, Expression):
        return {str(factor_pool): factor_pool}, factor_pool.fields
    if isinstance(factor_pool, dict):
        dependency = set()
        for name, expression in factor_pool.items():
            if not isinstance(name, str):
                raise ValueError(f"factor_name {name} should be string")
            if not isinstance(expression, (Expression, str)):
                raise ValueError(f"expression {name} should be an Expression or a string")
            dependency.update(
                expression.fields if isinstance(expression, Expression) else [expression]
            )
        return dict(factor_pool), sorted(dependency)
    if isinstance(factor_pool, (list, tuple)):
        expressions = {}
        dependency = set()
        for factor in factor_pool:
            if isinstance(factor, str):
                expressions[factor] = factor
                dependency.add(factor)
            elif isinstance(factor, Expression):
                expressions[str(factor)] = factor
                dependency.update(factor.fields)
            else:
                raise ValueError(f"Invalid factor: {factor}")
        return expressions, sorted(dependency)
    raise ValueError(f"{factor_pool} is not in valid format as factors")


class Transformer:
    def __init__(self, expressions):
        expression_dict, expression_dependency = factor_translator(expressions)
        if expression_dict:
            self.names = sorted(expression_dict)
            self.expressions = [expression_dict[name] for name in self.names]
            self.dependency = expression_dependency
        else:
            self.names = []
            self.expressions = []
            self.dependency = []

    def transform(self, group_name, data) -> pa.Table:
        table = as_arrow(data)
        if not self.names or table.num_rows == 0:
            return empty_table(self.names)
        requires_order = any(
            isinstance(expression, Expression) and expression.requires_order
            for expression in self.expressions
        )
        if requires_order:
            if "trade_date" not in table.column_names:
                raise ValueError("Ordered factor expressions require a trade_date column")
            if group_name not in table.column_names:
                raise ValueError(f"Ordered factor expressions require a {group_name} column")
            grouping_names = set().union(
                *[
                    expression.group_fields
                    for expression in self.expressions
                    if isinstance(expression, Expression)
                ]
            )
            required = {group_name, "trade_date"}.union(grouping_names)
            table = drop_nulls(table, sorted(required), drop_nan=False)
            schema = pa.schema(
                [
                    pa.field(
                        field.name,
                        field.type,
                        nullable=False if field.name in required else field.nullable,
                        metadata=field.metadata,
                    )
                    for field in table.schema
                ]
            )
            table = pa.Table.from_arrays(table.columns, schema=schema)
            order = [("trade_date", "ascending")]
            if group_name in table.column_names:
                order.append((group_name, "ascending"))
            table = table.sort_by(order)
            source = cf.table_input(
                "input",
                schema=table.schema,
                entity_by=(group_name,),
                event_time="trade_date",
                sequence_by=("trade_date",),
            )
        else:
            source = None

        temporary_names = [f"__alpha_factor_{index}" for index in range(len(self.names))]
        if set(temporary_names).intersection(table.column_names):
            raise ValueError("Input contains reserved factor column names")

        def build(input_table):
            features = {}
            for temporary, expression in zip(temporary_names, self.expressions):
                features[temporary] = (
                    expression.bind(input_table)
                    if isinstance(expression, Expression)
                    else input_table[expression]
                )
            return input_table.with_columns(features).select(*temporary_names)

        if source is None:
            result = cf.compute(table, build)
        else:
            result = build(source).collect(table)
        return pa.Table.from_arrays([result[name] for name in temporary_names], names=self.names)
