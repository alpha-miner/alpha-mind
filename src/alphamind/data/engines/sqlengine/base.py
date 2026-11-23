from __future__ import annotations

import os
from typing import List

import calc_flow as cf
import pyarrow as pa
import pyarrow.compute as pc
import sqlalchemy as sa
from sqlalchemy import Table, and_, join, outerjoin, select

from alphamind.data.arrow import (
    as_arrow,
    column_values,
    drop_duplicates,
    drop_nulls,
    read_sql,
    replace_inf,
    timestamp_scalar,
    to_numpy,
    with_column,
)
from alphamind.data.calendar import advanceDateByCalendar
from alphamind.data.engines.utilities import _map_factors
from alphamind.data.processing import factor_processing
from alphamind.data.transformer import Transformer
from alphamind.portfolio.riskmodel import FactorRiskModel


class SqlEngineBase:
    models = None
    metadata = None
    total_risk_factors = []
    percent_returns = True
    index_codes = {}

    def __init__(self, db_url: str, factor_tables: List[str] = None):
        self._engine = sa.create_engine(db_url)
        names = (
            factor_tables
            if factor_tables is not None
            else [
                name.strip()
                for name in os.environ.get("FACTOR_TABLES", "").split(",")
                if name.strip()
            ]
        )
        self._factor_tables = [self._factor_table(name) for name in names]
        existing = {table.name for table in self._factor_tables}
        for table in (self.models.Market.__table__, self.models.RiskExposure.__table__):
            if table.name not in existing:
                self._factor_tables.append(table)

    @property
    def engine(self):
        return self._engine

    @property
    def universe_table(self):
        return self.models.Universe

    def close(self):
        self._engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _factor_table(self, name):
        name = name.strip()
        return (
            self.metadata.tables[name]
            if name in self.metadata.tables
            else Table(name, self.metadata, autoload_with=self._engine)
        )

    def _read(self, query):
        return read_sql(query, self._engine)

    def _flag_condition(self, table):
        return table.flag == 1 if hasattr(table, "flag") else sa.true()

    def _date_condition(self, column, start_date, end_date, dates):
        return column.in_(dates) if dates is not None else column.between(start_date, end_date)

    def _map_index(self, benchmark):
        return self.index_codes.get(benchmark, benchmark)

    def fetch_factors_meta(self):
        return self._read(select(self.models.FactorMaster))

    def fetch_risk_meta(self):
        return self._read(select(self.models.RiskMaster))

    def fetch_codes(self, ref_date: str, universe):
        table = universe.query(self, ref_date, ref_date)
        return sorted(table["code"].to_pylist())

    def fetch_codes_range(self, universe, start_date=None, end_date=None, dates=None):
        return universe.query(self, start_date, end_date, dates)

    def _forward_returns(self, data, horizon, offset):
        table = as_arrow(data)
        first = offset + 1
        last = first + horizon
        scale = 100.0 if self.percent_returns else 1.0
        query = f"""
            SELECT trade_date, code,
                   SUM(LN(1 + "chgPct" / {scale})) OVER (
                       PARTITION BY code ORDER BY trade_date
                       ROWS BETWEEN {first} FOLLOWING AND {last} FOLLOWING
                   ) AS dx,
                   COUNT("chgPct") OVER (
                       PARTITION BY code ORDER BY trade_date
                       ROWS BETWEEN {first} FOLLOWING AND {last} FOLLOWING
                   ) AS samples
            FROM input
            ORDER BY trade_date, code
        """
        result = cf.compute(table, lambda input_table: input_table.sql(query))
        result = result.filter(pc.equal(result["samples"], horizon + 1))
        return result.select(["trade_date", "code", "dx"])

    def _subtract_benchmark(self, data, benchmark_data):
        benchmark = benchmark_data.select(["trade_date", "dx"]).rename_columns(
            ["trade_date", "benchmark_dx"]
        )
        result = data.join(benchmark, keys="trade_date", join_type="left outer")
        values = column_values(result, "dx") - column_values(result, "benchmark_dx")
        return with_column(result, "dx", values).drop(["benchmark_dx"])

    def fetch_dx_return(
        self,
        ref_date,
        codes,
        expiry_date=None,
        horizon=0,
        offset=0,
        neutralized_risks=None,
        pre_process=None,
        post_process=None,
        benchmark=None,
    ):
        end_date = expiry_date or advanceDateByCalendar(
            "china.sse", ref_date, f"{1 + horizon + offset}b"
        ).strftime("%Y-%m-%d")
        market = self.models.Market
        query = (
            select(market.trade_date, market.code.label("code"), market.chgPct.label("chgPct"))
            .where(
                and_(
                    market.trade_date.between(ref_date, end_date),
                    market.code.in_(codes),
                    self._flag_condition(market),
                )
            )
            .order_by(market.trade_date, market.code)
        )
        result = self._forward_returns(self._read(query), horizon, offset)
        result = result.filter(pc.equal(result["trade_date"], timestamp_scalar(ref_date)))
        if benchmark is not None:
            index_return = self.fetch_dx_return_index(
                ref_date, benchmark, expiry_date=expiry_date, horizon=horizon, offset=offset
            )
            if index_return.num_rows:
                values = column_values(result, "dx") - index_return["dx"][0].as_py()
                result = with_column(result, "dx", values)
        if neutralized_risks:
            risk = self.fetch_risk_model(ref_date, codes)[1]
            result = result.join(risk, keys="code", join_type="inner")
            result = drop_nulls(result)
            values = factor_processing(
                to_numpy(result, ["dx"], dtype=float),
                pre_process=pre_process,
                risk_factors=to_numpy(result, neutralized_risks, dtype=float),
                post_process=post_process,
            )
            result = with_column(result, "dx", values[:, 0])
        return result.select(["code", "dx"])

    def fetch_dx_return_index(self, ref_date, index_code, expiry_date=None, horizon=0, offset=0):
        end_date = expiry_date or advanceDateByCalendar(
            "china.sse", ref_date, f"{1 + horizon + offset}b"
        ).strftime("%Y-%m-%d")
        market = self.models.IndexMarket
        index_code = self._map_index(index_code)
        query = (
            select(market.trade_date, market.indexCode.label("code"), market.chgPct.label("chgPct"))
            .where(
                and_(
                    market.trade_date.between(ref_date, end_date),
                    market.indexCode == index_code,
                    self._flag_condition(market),
                )
            )
            .order_by(market.trade_date)
        )
        result = self._forward_returns(self._read(query), horizon, offset)
        result = result.filter(pc.equal(result["trade_date"], timestamp_scalar(ref_date)))
        return result.select(["code", "dx"])

    def fetch_dx_return_range(
        self,
        universe,
        start_date=None,
        end_date=None,
        dates=None,
        horizon=0,
        offset=0,
        benchmark=None,
    ):
        if dates is not None:
            start_date, end_date = dates[0], dates[-1]
        universe_data = self.fetch_codes_range(universe, start_date, end_date, dates)
        query_end = advanceDateByCalendar(
            "china.sse", end_date, f"{1 + horizon + offset}b"
        ).strftime("%Y-%m-%d")
        market = self.models.Market
        codes = pc.unique(universe_data["code"]).to_pylist()
        query = (
            select(market.trade_date, market.code.label("code"), market.chgPct.label("chgPct"))
            .where(
                and_(
                    market.trade_date.between(start_date, query_end),
                    market.code.in_(codes),
                    self._flag_condition(market),
                )
            )
            .order_by(market.trade_date, market.code)
        )
        result = self._forward_returns(self._read(query), horizon, offset)
        result = result.join(universe_data, keys=["trade_date", "code"], join_type="inner")
        if benchmark is not None:
            benchmark_data = self.fetch_dx_return_index_range(
                benchmark, start_date, end_date, dates, horizon, offset
            )
            result = self._subtract_benchmark(result, benchmark_data)
        return result.sort_by([("trade_date", "ascending"), ("code", "ascending")])

    def fetch_dx_return_index_range(
        self, index_code, start_date=None, end_date=None, dates=None, horizon=0, offset=0
    ):
        if dates is not None:
            start_date, end_date = dates[0], dates[-1]
        query_end = advanceDateByCalendar(
            "china.sse", end_date, f"{1 + horizon + offset}b"
        ).strftime("%Y-%m-%d")
        market = self.models.IndexMarket
        index_code = self._map_index(index_code)
        query = (
            select(market.trade_date, market.indexCode.label("code"), market.chgPct.label("chgPct"))
            .where(
                and_(
                    market.trade_date.between(start_date, query_end),
                    market.indexCode == index_code,
                    self._flag_condition(market),
                )
            )
            .order_by(market.trade_date)
        )
        result = self._forward_returns(self._read(query), horizon, offset)
        if dates is not None:
            values = pa.array(
                [timestamp_scalar(value).value for value in dates],
                type=pa.timestamp("us", tz="UTC"),
            )
            result = result.filter(pc.is_in(result["trade_date"], value_set=values))
        return result

    def _factor_query(self, transformer, codes=None, start_date=None, end_date=None, dates=None):
        factor_cols = _map_factors(transformer.dependency, self._factor_tables)
        market = self.models.Market
        source = market
        joined = {market.__table__.name}
        for table in set(factor_cols.values()):
            if table.name in joined:
                continue
            code_column = (
                table.columns["security_code"]
                if "security_code" in table.columns
                else table.columns["code"]
            )
            conditions = [
                market.trade_date == table.columns["trade_date"],
                market.code == code_column,
                self._flag_condition(market),
            ]
            if hasattr(table, "flag"):
                conditions.append(table.columns["flag"] == 1)
            source = outerjoin(source, table, and_(*conditions))
            joined.add(table.name)
        columns = [
            market.trade_date,
            market.code.label("code"),
            market.chgPct.label("chgPct"),
            market.secShortName.label("secShortName"),
        ]
        columns.extend(table.columns[name] for name, table in factor_cols.items())
        conditions = [
            self._date_condition(market.trade_date, start_date, end_date, dates),
            self._flag_condition(market),
        ]
        if codes is not None:
            conditions.append(market.code.in_(list(codes)))
        query = select(*columns).select_from(source).where(and_(*conditions)).distinct()
        return self._read(query)

    @staticmethod
    def _transformed_factors(transformer, data):
        table = drop_duplicates(
            replace_inf(data).sort_by([("trade_date", "ascending"), ("code", "ascending")]),
            ["trade_date", "code"],
        )
        table = drop_nulls(table, ["trade_date", "code"], drop_nan=False)
        transformed = transformer.transform("code", table)
        for name in ("trade_date", "code", "chgPct", "secShortName"):
            transformed = with_column(transformed, name, table[name])
        return transformed.select(
            ["trade_date", "code"] + transformer.names + ["chgPct", "secShortName"]
        )

    def fetch_factor(self, ref_date, factors, codes, warm_start=0):
        transformer = factors if isinstance(factors, Transformer) else Transformer(factors)
        start_date = advanceDateByCalendar("china.sse", ref_date, f"-{warm_start}b").strftime(
            "%Y-%m-%d"
        )
        data = self._factor_query(
            transformer, codes=codes, start_date=start_date, end_date=ref_date
        )
        result = self._transformed_factors(transformer, data)
        return result.filter(pc.equal(result["trade_date"], timestamp_scalar(ref_date)))

    def fetch_factor_range(
        self, universe, factors, start_date=None, end_date=None, dates=None, external_data=None
    ):
        transformer = factors if isinstance(factors, Transformer) else Transformer(factors)
        universe_data = self.fetch_codes_range(universe, start_date, end_date, dates)
        codes = pc.unique(universe_data["code"]).to_pylist()
        data = self._factor_query(
            transformer, codes=codes, start_date=start_date, end_date=end_date, dates=dates
        )
        if external_data is not None:
            data = data.join(
                as_arrow(external_data), keys=["trade_date", "code"], join_type="inner"
            )
        result = self._transformed_factors(transformer, data)
        return result.join(universe_data, keys=["trade_date", "code"], join_type="inner")

    def fetch_factor_range_forward(
        self, universe, factors, start_date=None, end_date=None, dates=None
    ):
        transformer = factors if isinstance(factors, Transformer) else Transformer(factors)
        result = self.fetch_factor_range(universe, transformer, start_date, end_date, dates)
        name = transformer.names[0]
        quoted = name.replace('"', '""')
        query = f'''
            SELECT trade_date, code,
                   LEAD("{quoted}") OVER (PARTITION BY code ORDER BY trade_date) AS dx
            FROM input
            ORDER BY trade_date, code
        '''
        result = cf.compute(result, lambda input_table: input_table.sql(query))
        return drop_nulls(result)

    def _industry_columns(self, level):
        code_name = (
            f"industry_code{level}"
            if hasattr(self.models.Industry, f"industry_code{level}")
            else f"industryID{level}"
        )
        category_name = f"industry_name{level}"
        if not hasattr(self.models.Industry, category_name):
            category_name = f"industryName{level}"
        return code_name, category_name

    def fetch_industry(self, ref_date, codes=None, category="sw", level=1):
        industry = self.models.Industry
        code_name, category_name = self._industry_columns(level)
        conditions = [industry.trade_date == ref_date, self._flag_condition(industry)]
        if codes is not None:
            conditions.append(industry.code.in_(list(codes)))
        if hasattr(industry, "industry"):
            conditions.append(industry.industry == self.industry_categories[category])
        query = (
            select(
                industry.code.label("code"),
                getattr(industry, code_name).label("industry_code"),
                getattr(industry, category_name).label("industry"),
            )
            .where(and_(*conditions))
            .distinct()
        )
        result = drop_nulls(self._read(query))
        return drop_duplicates(result, ["code"])

    def fetch_industry_range(
        self, universe, start_date=None, end_date=None, dates=None, category="sw", level=1
    ):
        industry = self.models.Industry
        code_name, category_name = self._industry_columns(level)
        universe_data = self.fetch_codes_range(universe, start_date, end_date, dates)
        conditions = [
            self._date_condition(industry.trade_date, start_date, end_date, dates),
            industry.code.in_(pc.unique(universe_data["code"]).to_pylist()),
            self._flag_condition(industry),
        ]
        if hasattr(industry, "industry"):
            conditions.append(industry.industry == self.industry_categories[category])
        query = (
            select(
                industry.trade_date,
                industry.code.label("code"),
                getattr(industry, code_name).label("industry_code"),
                getattr(industry, category_name).label("industry"),
            )
            .where(and_(*conditions))
            .distinct()
        )
        result = drop_nulls(self._read(query))
        return result.join(universe_data, keys=["trade_date", "code"], join_type="inner")

    @staticmethod
    def _industry_matrix(data):
        table = with_column(data, "industry_name", data["industry"])
        industries = sorted(pc.unique(table["industry"]).to_pylist())
        for name in industries:
            table = table.append_column(
                str(name), pc.cast(pc.equal(table["industry"], name), pa.float64())
            )
        return table.drop(["industry_code", "industry"])

    def fetch_industry_matrix(self, ref_date, codes=None, category="sw", level=1):
        return self._industry_matrix(self.fetch_industry(ref_date, codes, category, level))

    def fetch_industry_matrix_range(
        self, universe, start_date=None, end_date=None, dates=None, category="sw", level=1
    ):
        result = self._industry_matrix(
            self.fetch_industry_range(universe, start_date, end_date, dates, category, level)
        )
        return drop_duplicates(result, ["trade_date", "code"])

    def _risk_tables(self, risk_model):
        return {
            "day": (self.models.RiskCovDay, self.models.SpecificRiskDay),
            "short": (self.models.RiskCovShort, self.models.SpecificRiskShort),
            "long": (self.models.RiskCovLong, self.models.SpecificRiskLong),
        }[risk_model]

    def _risk_covariance(
        self, risk_model, start_date=None, end_date=None, dates=None, ref_date=None
    ):
        table = self._risk_tables(risk_model)[0]
        columns = [table.FactorID, table.Factor] + [
            table.__table__.columns[name] for name in self.total_risk_factors
        ]
        if ref_date is None:
            columns.insert(0, table.trade_date)
        if ref_date is not None:
            condition = table.trade_date == ref_date
        else:
            condition = self._date_condition(table.trade_date, start_date, end_date, dates)
        query = select(*columns).where(and_(condition, self._flag_condition(table)))
        result = self._read(query)
        keys = ["FactorID"] if ref_date is not None else ["trade_date", "FactorID"]
        return result.sort_by([(key, "ascending") for key in keys])

    def _risk_exposure(
        self,
        risk_model,
        codes,
        start_date=None,
        end_date=None,
        dates=None,
        ref_date=None,
        excluded=None,
    ):
        risk = self.models.RiskExposure
        special = self._risk_tables(risk_model)[1]
        excluded = set(excluded or [])
        names = [name for name in self.total_risk_factors if name not in excluded]
        source = join(
            risk,
            special,
            and_(
                risk.code == special.code,
                risk.trade_date == special.trade_date,
                self._flag_condition(special),
            ),
        )
        columns = [risk.code.label("code"), special.SRISK.label("srisk")]
        if ref_date is None:
            columns.insert(0, risk.trade_date)
        columns.extend(risk.__table__.columns[name] for name in names)
        date_condition = (
            risk.trade_date == ref_date
            if ref_date is not None
            else self._date_condition(risk.trade_date, start_date, end_date, dates)
        )
        query = (
            select(*columns)
            .select_from(source)
            .where(and_(date_condition, risk.code.in_(list(codes)), self._flag_condition(risk)))
            .distinct()
        )
        result = drop_nulls(self._read(query))
        return drop_duplicates(result, ["code"] if ref_date is not None else ["trade_date", "code"])

    @staticmethod
    def _factor_risk_model(covariance, exposure):
        for name in covariance.column_names:
            if name not in ("Factor", "FactorID", "trade_date"):
                covariance = with_column(
                    covariance, name, column_values(covariance, name) / 10000.0
                )
        special = pa.table(
            {
                "code": exposure["code"],
                "idsync": column_values(exposure, "srisk") ** 2 / 10000.0,
            }
        )
        return FactorRiskModel(covariance, exposure, special)

    def fetch_risk_model(self, ref_date, codes, risk_model="short", excluded=None, model_type=None):
        covariance = self._risk_covariance(risk_model, ref_date=ref_date)
        exposure = self._risk_exposure(risk_model, codes, ref_date=ref_date, excluded=excluded)
        if model_type is None:
            return covariance, exposure
        if model_type == "factor":
            return self._factor_risk_model(covariance, exposure), covariance, exposure
        raise ValueError(f"Unknown risk model type: {model_type}")

    def fetch_risk_model_range(
        self,
        universe,
        start_date=None,
        end_date=None,
        dates=None,
        risk_model="short",
        excluded=None,
        model_type=None,
    ):
        covariance = self._risk_covariance(risk_model, start_date, end_date, dates)
        universe_data = self.fetch_codes_range(universe, start_date, end_date, dates)
        exposure = self._risk_exposure(
            risk_model,
            pc.unique(universe_data["code"]).to_pylist(),
            start_date,
            end_date,
            dates,
            excluded=excluded,
        )
        exposure = exposure.join(universe_data, keys=["trade_date", "code"], join_type="inner")
        if model_type is None:
            return covariance, exposure
        if model_type != "factor":
            raise ValueError(f"Unknown risk model type: {model_type}")
        models = {}
        for ref_date in pc.unique(covariance["trade_date"]).to_pylist():
            cov = covariance.filter(pc.equal(covariance["trade_date"], ref_date))
            exp = exposure.filter(pc.equal(exposure["trade_date"], ref_date))
            models[ref_date] = self._factor_risk_model(cov, exp)
        return models, covariance, exposure

    def fetch_benchmark(self, ref_date, benchmark, codes=None):
        component = self.models.IndexComponent
        benchmark = self._map_index(benchmark)
        if self.use_index_weight:
            weight = self.models.IndexWeight
            source = join(
                component,
                weight,
                and_(
                    component.trade_date == weight.trade_date,
                    component.indexSymbol == weight.indexSymbol,
                    component.symbol == weight.symbol,
                    self._flag_condition(weight),
                ),
            )
            weight_column = weight.weight
        else:
            source = component
            weight_column = component.weight
        query = (
            select(component.code.label("code"), (weight_column / 100.0).label("weight"))
            .select_from(source)
            .where(
                and_(
                    component.trade_date == ref_date,
                    component.indexCode == benchmark,
                    self._flag_condition(component),
                )
            )
            .distinct()
        )
        result = drop_duplicates(self._read(query), ["code"])
        if codes is None:
            return result
        weights = dict(zip(result["code"].to_pylist(), column_values(result, "weight")))
        return pa.table({"code": list(codes), "weight": [weights.get(code, 0.0) for code in codes]})

    def fetch_benchmark_range(self, benchmark, start_date=None, end_date=None, dates=None):
        component = self.models.IndexComponent
        benchmark = self._map_index(benchmark)
        if self.use_index_weight:
            weight = self.models.IndexWeight
            source = join(
                component,
                weight,
                and_(
                    component.trade_date == weight.trade_date,
                    component.indexSymbol == weight.indexSymbol,
                    component.symbol == weight.symbol,
                    self._flag_condition(weight),
                ),
            )
            weight_column = weight.weight
        else:
            source = component
            weight_column = component.weight
        query = (
            select(
                component.trade_date,
                component.code.label("code"),
                (weight_column / 100.0).label("weight"),
            )
            .select_from(source)
            .where(
                and_(
                    self._date_condition(component.trade_date, start_date, end_date, dates),
                    component.indexCode == benchmark,
                    self._flag_condition(component),
                )
            )
            .distinct()
        )
        return drop_duplicates(self._read(query), ["trade_date", "code"])

    def fetch_trade_status(self, ref_date, codes, offset=0):
        target_date = advanceDateByCalendar("china.sse", ref_date, f"{offset}b").strftime(
            "%Y-%m-%d"
        )
        market = self.models.Market
        query = (
            select(market.trade_date, market.code.label("code"), market.isOpen.label("is_open"))
            .where(
                and_(
                    market.trade_date == target_date,
                    market.code.in_(list(codes)),
                    self._flag_condition(market),
                )
            )
            .order_by(market.code)
        )
        return self._read(query)

    def fetch_trade_status_range(
        self, universe, start_date=None, end_date=None, dates=None, offset=0
    ):
        if dates is not None:
            start_date, end_date = dates[0], dates[-1]
        universe_data = self.fetch_codes_range(universe, start_date, end_date, dates)
        target_end = advanceDateByCalendar("china.sse", end_date, f"{offset}b").strftime("%Y-%m-%d")
        market = self.models.Market
        query = select(market.trade_date, market.code.label("code"), market.isOpen).where(
            and_(
                market.trade_date.between(start_date, target_end),
                market.code.in_(pc.unique(universe_data["code"]).to_pylist()),
                self._flag_condition(market),
            )
        )
        data = self._read(query)
        query_sql = f"""
            SELECT trade_date, code,
                   LEAD("is_open", {offset}) OVER (PARTITION BY code ORDER BY trade_date) AS is_open
            FROM input
        """
        result = cf.compute(data, lambda input_table: input_table.sql(query_sql))
        return drop_nulls(result).join(
            universe_data, keys=["trade_date", "code"], join_type="inner"
        )

    def fetch_data(
        self, ref_date, factors, codes, benchmark=None, risk_model="short", industry="sw"
    ):
        transformer = factors if isinstance(factors, Transformer) else Transformer(factors)
        factor_data = self.fetch_factor(ref_date, transformer, codes)
        result = {"factor": factor_data}
        if benchmark is not None:
            benchmark_data = self.fetch_benchmark(ref_date, benchmark)
            result["benchmark"] = benchmark_data
            factor_data = factor_data.join(benchmark_data, keys="code", join_type="left outer")
            factor_data = with_column(
                factor_data, "weight", pc.fill_null(factor_data["weight"], 0.0)
            )
        if risk_model is not None:
            excluded = set(self.total_risk_factors).intersection(transformer.dependency)
            risk_cov, risk_exp = self.fetch_risk_model(
                ref_date, codes, risk_model, sorted(excluded)
            )
            result["risk_cov"] = risk_cov
            factor_data = factor_data.join(risk_exp, keys="code", join_type="left outer")
        industry_info = self.fetch_industry(ref_date, codes, industry)
        result["factor"] = factor_data.join(industry_info, keys="code", join_type="inner")
        return result

    def fetch_data_range(
        self,
        universe,
        factors,
        start_date=None,
        end_date=None,
        dates=None,
        benchmark=None,
        risk_model="short",
        industry="sw",
        external_data=None,
    ):
        transformer = factors if isinstance(factors, Transformer) else Transformer(factors)
        factor_data = self.fetch_factor_range(
            universe, transformer, start_date, end_date, dates, external_data
        )
        result = {"factor": factor_data}
        keys = ["trade_date", "code"]
        if benchmark is not None:
            benchmark_data = self.fetch_benchmark_range(benchmark, start_date, end_date, dates)
            result["benchmark"] = benchmark_data
            factor_data = factor_data.join(benchmark_data, keys=keys, join_type="left outer")
            factor_data = with_column(
                factor_data, "weight", pc.fill_null(factor_data["weight"], 0.0)
            )
        if risk_model is not None:
            excluded = set(self.total_risk_factors).intersection(transformer.dependency)
            risk_cov, risk_exp = self.fetch_risk_model_range(
                universe, start_date, end_date, dates, risk_model, sorted(excluded)
            )
            result["risk_cov"] = risk_cov
            factor_data = factor_data.join(risk_exp, keys=keys, join_type="left outer")
        industry_info = self.fetch_industry_range(universe, start_date, end_date, dates, industry)
        result["factor"] = factor_data.join(industry_info, keys=keys, join_type="inner")
        return result
