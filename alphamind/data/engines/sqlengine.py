# -*- coding: utf-8 -*-
"""
Created on 2017-7-7

@author: cheng.li
"""

from typing import Iterable
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
import pandas as pd
import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy import select, and_, outerjoin, join, delete, insert
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import bindparam
from alphamind.data.engines.universe import Universe
from alphamind.data.dbmodel.models import FactorMaster
from alphamind.data.dbmodel.models import FactorLog
from alphamind.data.dbmodel.models import Strategy
from alphamind.data.dbmodel.models import IndexComponent
from alphamind.data.dbmodel.models import Industry
from alphamind.data.dbmodel.models import Experimental
from alphamind.data.dbmodel.models import RiskMaster
from alphamind.data.dbmodel.models import FullFactor
from alphamind.data.dbmodel.models import Models
from alphamind.data.dbmodel.models import Market
from alphamind.data.dbmodel.models import IndexMarket
from alphamind.data.dbmodel.models import Universe as UniverseTable
from alphamind.data.dbmodel.models import Formulas
from alphamind.data.dbmodel.models import DailyPortfoliosSchedule
from alphamind.data.dbmodel.models import Performance
from alphamind.data.dbmodel.models import Positions
from alphamind.data.transformer import Transformer
from alphamind.model.loader import load_model
from alphamind.formula.utilities import encode_formula
from alphamind.formula.utilities import decode_formula
from alphamind.data.engines.utilities import _map_factors
from alphamind.data.engines.utilities import _map_industry_category
from alphamind.data.engines.utilities import _map_risk_model_table
from alphamind.data.engines.utilities import factor_tables
from PyFin.api import advanceDateByCalendar

risk_styles = ['BETA',
               'MOMENTUM',
               'SIZE',
               'EARNYILD',
               'RESVOL',
               'GROWTH',
               'BTOP',
               'LEVERAGE',
               'LIQUIDTY',
               'SIZENL']

industry_styles = [
    'Bank',
    'RealEstate',
    'Health',
    'Transportation',
    'Mining',
    'NonFerMetal',
    'HouseApp',
    'LeiService',
    'MachiEquip',
    'BuildDeco',
    'CommeTrade',
    'CONMAT',
    'Auto',
    'Textile',
    'FoodBever',
    'Electronics',
    'Computer',
    'LightIndus',
    'Utilities',
    'Telecom',
    'AgriForest',
    'CHEM',
    'Media',
    'IronSteel',
    'NonBankFinan',
    'ELECEQP',
    'AERODEF',
    'Conglomerates'
]

macro_styles = ['COUNTRY']

total_risk_factors = risk_styles + industry_styles + macro_styles

DEFAULT_URL = 'postgresql+psycopg2://postgres:A12345678!@10.63.6.220/alpha'

DAILY_RETURN_OFFSET = 0


class SqlEngine(object):
    def __init__(self,
                 db_url: str = None):
        if db_url:
            self.engine = sa.create_engine(db_url)
        else:
            self.engine = sa.create_engine(DEFAULT_URL)

        self.session = None
        self.create_session()

        if self.engine.name == 'mssql':
            self.ln_func = func.log
        else:
            self.ln_func = func.ln

    def create_session(self):
        db_session = orm.sessionmaker(bind=self.engine)
        self.session = db_session()

    def fetch_factors_meta(self) -> pd.DataFrame:
        query = self.session.query(FactorMaster)
        return pd.read_sql(query.statement, query.session.bind)

    def fetch_factor_coverage(self) -> pd.DataFrame:
        query = self.session.query(FactorLog)
        return pd.read_sql(query.statement, query.session.bind)

    def fetch_risk_meta(self) -> pd.DataFrame:
        query = self.session.query(RiskMaster)
        return pd.read_sql(query.statement, query.session.bind)

    def fetch_strategy(self, ref_date: str, strategy: str) -> pd.DataFrame():
        query = select([Strategy.strategyName, Strategy.factor, Strategy.weight]).where(
            and_(
                Strategy.trade_date == ref_date,
                Strategy.strategyName == strategy
            )
        )

        return pd.read_sql(query, self.session.bind)

    def fetch_strategy_names(self):
        query = select([Strategy.strategyName]).distinct()
        cursor = self.engine.execute(query)
        strategy_names = {s[0] for s in cursor.fetchall()}
        return strategy_names

    def fetch_codes(self, ref_date: str, universe: Universe) -> List[int]:
        cond = universe.query(ref_date)
        query = select([UniverseTable.trade_date, UniverseTable.code]).distinct().where(cond)
        cursor = self.engine.execute(query)
        codes_set = {c[1] for c in cursor.fetchall()}
        return sorted(codes_set)

    def fetch_codes_range(self,
                          universe: Universe,
                          start_date: str = None,
                          end_date: str = None,
                          dates: Iterable[str] = None) -> pd.DataFrame:
        cond = universe.query_range(start_date, end_date, dates)
        query = select([UniverseTable.trade_date, UniverseTable.code]).distinct().where(cond)
        return pd.read_sql(query, self.engine)

    def fetch_dx_return(self,
                        ref_date: str,
                        codes: Iterable[int],
                        expiry_date: str = None,
                        horizon: int = 0,
                        offset: int = 0) -> pd.DataFrame:
        start_date = ref_date

        if not expiry_date:
            end_date = advanceDateByCalendar('china.sse', ref_date,
                                             str(1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime('%Y%m%d')
        else:
            end_date = expiry_date

        stats = func.sum(self.ln_func(1. + Market.chgPct)).over(
            partition_by=Market.code,
            order_by=Market.trade_date,
            rows=(1 + DAILY_RETURN_OFFSET + offset, 1 + horizon + DAILY_RETURN_OFFSET + offset)).label('dx')

        query = select([Market.trade_date, Market.code, stats]).where(
            and_(
                Market.trade_date.between(start_date, end_date),
                Market.code.in_(codes)
            )
        )

        df = pd.read_sql(query, self.session.bind).dropna()
        df = df[df.trade_date == ref_date]
        return df[['code', 'dx']]

    def fetch_dx_return_range(self,
                              universe,
                              start_date: str = None,
                              end_date: str = None,
                              dates: Iterable[str] = None,
                              horizon: int = 0,
                              offset: int = 0) -> pd.DataFrame:

        if dates:
            start_date = dates[0]
            end_date = dates[-1]

        end_date = advanceDateByCalendar('china.sse', end_date,
                                         str(1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime('%Y-%m-%d')

        codes = universe.query(self, start_date, end_date, dates)

        stats = func.sum(self.ln_func(1. + Market.chgPct)).over(
            partition_by=Market.code,
            order_by=Market.trade_date,
            rows=(1 + offset + DAILY_RETURN_OFFSET, 1 + horizon + offset + DAILY_RETURN_OFFSET)).label('dx')

        query = select([Market.trade_date, Market.code, stats]) \
            .where(
            and_(Market.trade_date.in_(codes.trade_date.dt.strftime('%Y-%m-%d').unique()),
                 Market.code.in_(codes.code.tolist()))
        )

        df = pd.read_sql(query, self.session.bind).dropna()
        df = pd.merge(df, codes, how='inner', on=['trade_date', 'code'])

        if dates:
            df = df[df.trade_date.isin(dates)]
        return df

    def fetch_dx_return_index(self,
                              ref_date: str,
                              index_code: int,
                              expiry_date: str = None,
                              horizon: int = 0,
                              offset: int = 0) -> pd.DataFrame:
        start_date = ref_date

        if not expiry_date:
            end_date = advanceDateByCalendar('china.sse', ref_date,
                                             str(1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime('%Y%m%d')
        else:
            end_date = expiry_date

        stats = func.sum(self.ln_func(1. + IndexMarket.chgPct)).over(
            partition_by=IndexMarket.indexCode,
            order_by=IndexMarket.trade_date,
            rows=(1 + DAILY_RETURN_OFFSET + offset, 1 + horizon + DAILY_RETURN_OFFSET + offset)).label('dx')

        query = select([IndexMarket.trade_date, IndexMarket.indexCode.label('code'), stats]).where(
            and_(
                IndexMarket.trade_date.between(start_date, end_date),
                IndexMarket.indexCode == index_code
            )
        )

        df = pd.read_sql(query, self.session.bind).dropna()
        df = df[df.trade_date == ref_date]
        return df[['code', 'dx']]

    def fetch_dx_return_index_range(self,
                                    index_code,
                                    start_date: str = None,
                                    end_date: str = None,
                                    dates: Iterable[str] = None,
                                    horizon: int = 0,
                                    offset: int = 0) -> pd.DataFrame:

        if dates:
            start_date = dates[0]
            end_date = dates[-1]

        end_date = advanceDateByCalendar('china.sse', end_date,
                                         str(1 + horizon + offset + DAILY_RETURN_OFFSET) + 'b').strftime('%Y-%m-%d')

        stats = func.sum(self.ln_func(1. + IndexMarket.chgPct)).over(
            partition_by=IndexMarket.indexCode,
            order_by=IndexMarket.trade_date,
            rows=(1 + offset + DAILY_RETURN_OFFSET, 1 + horizon + offset + DAILY_RETURN_OFFSET)).label('dx')

        query = select([IndexMarket.trade_date, IndexMarket.indexCode.label('code'), stats]) \
            .where(
            and_(
                IndexMarket.trade_date.between(start_date, end_date),
                IndexMarket.indexCode == index_code
            )
        )

        df = pd.read_sql(query, self.session.bind).dropna()

        if dates:
            df = df[df.trade_date.isin(dates)]
        return df

    def fetch_factor(self,
                     ref_date: str,
                     factors: Iterable[object],
                     codes: Iterable[int],
                     warm_start: int = 0,
                     used_factor_tables=None) -> pd.DataFrame:

        if isinstance(factors, Transformer):
            transformer = factors
        else:
            transformer = Transformer(factors)

        dependency = transformer.dependency

        if used_factor_tables:
            factor_cols = _map_factors(dependency, used_factor_tables)
        else:
            factor_cols = _map_factors(dependency, factor_tables)

        start_date = advanceDateByCalendar('china.sse', ref_date, str(-warm_start) + 'b').strftime('%Y-%m-%d')
        end_date = ref_date

        big_table = FullFactor

        for t in set(factor_cols.values()):
            if t.__table__.name != FullFactor.__table__.name:
                big_table = outerjoin(big_table, t, and_(FullFactor.trade_date == t.trade_date,
                                                         FullFactor.code == t.code))

        query = select(
            [FullFactor.trade_date, FullFactor.code, FullFactor.isOpen] + list(factor_cols.keys())) \
            .select_from(big_table).where(and_(FullFactor.trade_date.between(start_date, end_date),
                                               FullFactor.code.in_(codes)))

        df = pd.read_sql(query, self.engine).sort_values(['trade_date', 'code']).set_index('trade_date')
        res = transformer.transform('code', df)

        for col in res.columns:
            if col not in set(['code', 'isOpen']) and col not in df.columns:
                df[col] = res[col].values

        df['isOpen'] = df.isOpen.astype(bool)
        df = df.loc[ref_date]
        df.index = list(range(len(df)))
        return df

    def fetch_factor_range(self,
                           universe: Universe,
                           factors: Union[Transformer, Iterable[object]],
                           start_date: str = None,
                           end_date: str = None,
                           dates: Iterable[str] = None,
                           external_data: pd.DataFrame = None,
                           used_factor_tables=None) -> pd.DataFrame:

        if isinstance(factors, Transformer):
            transformer = factors
        else:
            transformer = Transformer(factors)

        dependency = transformer.dependency

        if used_factor_tables:
            factor_cols = _map_factors(dependency, used_factor_tables)
        else:
            factor_cols = _map_factors(dependency, factor_tables)

        codes = universe.query(self, start_date, end_date, dates)

        big_table = FullFactor

        for t in set(factor_cols.values()):
            if t.__table__.name != FullFactor.__table__.name:
                if dates is not None:
                    big_table = outerjoin(big_table, t, and_(FullFactor.trade_date == t.trade_date,
                                                             FullFactor.code == t.code,
                                                             FullFactor.trade_date.in_(dates)))
                else:
                    big_table = outerjoin(big_table, t, and_(FullFactor.trade_date == t.trade_date,
                                                             FullFactor.code == t.code,
                                                             FullFactor.trade_date.between(start_date, end_date)))

        query = select(
            [FullFactor.trade_date, FullFactor.code, FullFactor.isOpen] + list(factor_cols.keys())) \
            .select_from(big_table).where(
            and_(FullFactor.trade_date.in_(codes.trade_date.dt.strftime('%Y-%m-%d').unique()),
                 FullFactor.code.in_(codes.code.tolist()))
        ).distinct()

        df = pd.read_sql(query, self.engine)
        df = pd.merge(df, codes, how='inner', on=['trade_date', 'code']).sort_values(['trade_date', 'code'])

        if external_data is not None:
            df = pd.merge(df, external_data, on=['trade_date', 'code']).dropna()

        df.set_index('trade_date', inplace=True)
        res = transformer.transform('code', df)

        for col in res.columns:
            if col not in set(['code', 'isOpen']) and col not in df.columns:
                df[col] = res[col].values

        df['isOpen'] = df.isOpen.astype(bool)
        return df.reset_index()

    def fetch_benchmark(self,
                        ref_date: str,
                        benchmark: int) -> pd.DataFrame:
        query = select([IndexComponent.code, (IndexComponent.weight / 100.).label('weight')]).where(
            and_(
                IndexComponent.trade_date == ref_date,
                IndexComponent.indexCode == benchmark
            )
        )

        return pd.read_sql(query, self.engine)

    def fetch_benchmark_range(self,
                              benchmark: int,
                              start_date: str = None,
                              end_date: str = None,
                              dates: Iterable[str] = None) -> pd.DataFrame:

        cond = IndexComponent.trade_date.in_(dates) if dates else IndexComponent.trade_date.between(start_date,
                                                                                                    end_date)

        query = select(
            [IndexComponent.trade_date, IndexComponent.code, (IndexComponent.weight / 100.).label('weight')]).where(
            and_(
                cond,
                IndexComponent.indexCode == benchmark
            )
        )
        return pd.read_sql(query, self.engine)

    def fetch_risk_model(self,
                         ref_date: str,
                         codes: Iterable[int],
                         risk_model: str = 'short',
                         excluded: Iterable[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        risk_cov_table, special_risk_col = _map_risk_model_table(risk_model)

        cov_risk_cols = [risk_cov_table.__table__.columns[f] for f in total_risk_factors]
        query = select([risk_cov_table.FactorID,
                        risk_cov_table.Factor]
                       + cov_risk_cols).where(
            risk_cov_table.trade_date == ref_date
        )
        risk_cov = pd.read_sql(query, self.engine).sort_values('FactorID')

        if excluded:
            risk_exposure_cols = [FullFactor.__table__.columns[f] for f in total_risk_factors if f not in set(excluded)]
        else:
            risk_exposure_cols = [FullFactor.__table__.columns[f] for f in total_risk_factors]

        query = select([FullFactor.code, special_risk_col] + risk_exposure_cols) \
            .where(and_(FullFactor.trade_date == ref_date, FullFactor.code.in_(codes))).distinct()

        risk_exp = pd.read_sql(query, self.engine)

        return risk_cov, risk_exp

    def fetch_risk_model_range(self,
                               universe: Universe,
                               start_date: str = None,
                               end_date: str = None,
                               dates: Iterable[str] = None,
                               risk_model: str = 'short',
                               excluded: Iterable[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        risk_cov_table, special_risk_col = _map_risk_model_table(risk_model)

        cov_risk_cols = [risk_cov_table.__table__.columns[f] for f in total_risk_factors]

        cond = risk_cov_table.trade_date.in_(dates) if dates else risk_cov_table.trade_date.between(start_date,
                                                                                                    end_date)
        query = select([risk_cov_table.trade_date,
                        risk_cov_table.FactorID,
                        risk_cov_table.Factor]
                       + cov_risk_cols).where(
            cond
        )

        risk_cov = pd.read_sql(query, self.engine).sort_values(['trade_date', 'FactorID'])

        if not excluded:
            excluded = []

        risk_exposure_cols = [FullFactor.__table__.columns[f] for f in total_risk_factors if f not in set(excluded)]

        codes = universe.query(self, start_date, end_date, dates)

        query = select(
            [FullFactor.trade_date, FullFactor.code, special_risk_col] + risk_exposure_cols).where(
            and_(FullFactor.trade_date.in_(codes.trade_date.dt.strftime('%Y-%m-%d').unique()),
                 FullFactor.code.in_(codes.code.tolist()))
        ).distinct()

        risk_exp = pd.read_sql(query, self.engine)
        risk_exp = pd.merge(risk_exp, codes, how='inner', on=['trade_date', 'code']).sort_values(['trade_date', 'code'])

        return risk_cov, risk_exp

    def fetch_industry(self,
                       ref_date: str,
                       codes: Iterable[int],
                       category: str = 'sw'):

        industry_category_name = _map_industry_category(category)

        query = select([Industry.code,
                        Industry.industryID1.label('industry_code'),
                        Industry.industryName1.label('industry')]).where(
            and_(
                Industry.trade_date == ref_date,
                Industry.code.in_(codes),
                Industry.industry == industry_category_name
            )
        ).distinct()

        return pd.read_sql(query, self.engine)

    def fetch_industry_range(self,
                             universe: Universe,
                             start_date: str = None,
                             end_date: str = None,
                             dates: Iterable[str] = None,
                             category: str = 'sw'):
        industry_category_name = _map_industry_category(category)
        codes = universe.query(self, start_date, end_date, dates)

        query = select([Industry.trade_date,
                        Industry.code,
                        Industry.industryID1.label('industry_code'),
                        Industry.industryName1.label('industry')]).where(
            and_(Industry.trade_date.in_(codes.trade_date.dt.strftime('%Y-%m-%d').unique()),
                 Industry.code.in_(codes.code.tolist()),
                 Industry.industry == industry_category_name)
        ).distinct()

        df = pd.read_sql(query, self.engine)
        df = pd.merge(df, codes, how='inner', on=['trade_date', 'code']).sort_values(['trade_date', 'code'])
        return df

    def fetch_data(self, ref_date: str,
                   factors: Iterable[str],
                   codes: Iterable[int],
                   benchmark: int = None,
                   risk_model: str = 'short',
                   industry: str = 'sw') -> Dict[str, pd.DataFrame]:

        total_data = {}

        transformer = Transformer(factors)
        factor_data = self.fetch_factor(ref_date,
                                        transformer,
                                        codes,
                                        used_factor_tables=[FullFactor, Experimental])

        if benchmark:
            benchmark_data = self.fetch_benchmark(ref_date, benchmark)
            total_data['benchmark'] = benchmark_data
            factor_data = pd.merge(factor_data, benchmark_data, how='left', on=['code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        if risk_model:
            excluded = list(set(total_risk_factors).intersection(transformer.dependency))
            risk_cov, risk_exp = self.fetch_risk_model(ref_date, codes, risk_model, excluded)
            factor_data = pd.merge(factor_data, risk_exp, how='left', on=['code'])
            total_data['risk_cov'] = risk_cov

        industry_info = self.fetch_industry(ref_date=ref_date,
                                            codes=codes,
                                            category=industry)

        factor_data = pd.merge(factor_data, industry_info, on=['code'])

        total_data['factor'] = factor_data
        return total_data

    def fetch_data_experimental(self, ref_date: str,
                                factors: Iterable[str],
                                codes: Iterable[int],
                                benchmark: int = None,
                                risk_model: str = 'short',
                                industry: str = 'sw') -> Dict[str, pd.DataFrame]:

        total_data = {}

        transformer = Transformer(factors)
        factor_data = self.fetch_factor(ref_date, transformer, codes, used_factor_tables=[Experimental])

        if benchmark:
            benchmark_data = self.fetch_benchmark(ref_date, benchmark)
            total_data['benchmark'] = benchmark_data
            factor_data = pd.merge(factor_data, benchmark_data, how='left', on=['code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        if risk_model:
            excluded = list(set(total_risk_factors).intersection(transformer.dependency))
            risk_cov, risk_exp = self.fetch_risk_model(ref_date, codes, risk_model, excluded)
            factor_data = pd.merge(factor_data, risk_exp, how='left', on=['code'])
            total_data['risk_cov'] = risk_cov

        industry_info = self.fetch_industry(ref_date=ref_date,
                                            codes=codes,
                                            category=industry)

        factor_data = pd.merge(factor_data, industry_info, on=['code'])

        total_data['factor'] = factor_data
        return total_data

    def fetch_data_range(self,
                         universe: Universe,
                         factors: Iterable[str],
                         start_date: str = None,
                         end_date: str = None,
                         dates: Iterable[str] = None,
                         benchmark: int = None,
                         risk_model: str = 'short',
                         industry: str = 'sw',
                         external_data: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:

        total_data = {}
        transformer = Transformer(factors)
        factor_data = self.fetch_factor_range(universe,
                                              transformer,
                                              start_date,
                                              end_date,
                                              dates,
                                              external_data=external_data)

        if benchmark:
            benchmark_data = self.fetch_benchmark_range(benchmark, start_date, end_date, dates)
            total_data['benchmark'] = benchmark_data
            factor_data = pd.merge(factor_data, benchmark_data, how='left', on=['trade_date', 'code'])
            factor_data['weight'] = factor_data['weight'].fillna(0.)

        if risk_model:
            excluded = list(set(total_risk_factors).intersection(transformer.dependency))
            risk_cov, risk_exp = self.fetch_risk_model_range(universe, start_date, end_date, dates, risk_model,
                                                             excluded)
            factor_data = pd.merge(factor_data, risk_exp, how='left', on=['trade_date', 'code'])
            total_data['risk_cov'] = risk_cov

        industry_info = self.fetch_industry_range(universe,
                                                  start_date=start_date,
                                                  end_date=end_date,
                                                  dates=dates,
                                                  category=industry)

        factor_data = pd.merge(factor_data, industry_info, on=['trade_date', 'code'])
        total_data['factor'] = factor_data
        return total_data

    def fetch_model(self,
                    ref_date=None,
                    model_type=None,
                    model_version=None,
                    is_primary=True,
                    model_id=None) -> pd.DataFrame:

        conditions = []

        if ref_date:
            conditions.append(Models.trade_date == ref_date)

        if model_id:
            conditions.append(Models.model_id == model_id)

        if model_type:
            conditions.append(Models.model_type == model_type)

        if model_version:
            conditions.append(Models.model_version == model_version)

        conditions.append(Models.is_primary == is_primary)

        query = select([Models]).where(and_(*conditions))

        model_df = pd.read_sql(query, self.engine)

        for i, model_desc in enumerate(model_df.model_desc):
            model_df.loc[i, 'model'] = load_model(model_desc)

        del model_df['model_desc']
        return model_df

    def insert_formula(self, formula_name, formula_obj):
        comment = str(formula_obj)
        dict_repr = encode_formula(formula=formula_obj)

        query = delete(Formulas).where(
            Formulas.formula == formula_name
        )

        self.engine.execute(query)

        query = insert(Formulas, values=dict(formula=formula_name,
                                             formula_desc=dict_repr,
                                             comment=comment))
        self.engine.execute(query)

    def load_formula(self, formula_name):
        query = select([Formulas]).where(
            Formulas.formula == formula_name
        )

        df = pd.read_sql(query, self.engine)

        if not df.empty:
            return decode_formula(df.loc[0, 'formula_desc']['desc'])

    def load_all_formulas(self):
        query = select([Formulas])

        df = pd.read_sql(query, self.engine, index_col='formula')

        if not df.empty:
            return pd.Series({name: decode_formula(df.loc[name, 'formula_desc']['desc']) for name in df.index})

    def insert_portfolio_schedule(self, df):
        query = insert(DailyPortfoliosSchedule).values(
            {
                DailyPortfoliosSchedule.portfolio_name: bindparam('portfolio_name'),
                DailyPortfoliosSchedule.trade_date: bindparam('trade_date')
            }
        )

        self.engine.execute(query, df.to_dict('record'))

    def upsert_performance(self, ref_date, df):
        build_types = df['type'].unique().tolist()
        universes = df['universe'].unique().tolist()
        benchmarks = df['benchmark'].unique().tolist()
        portfolios = df['portfolio'].unique().tolist()
        sources = df['source'].unique().tolist()

        query = delete(Performance).where(
            and_(
                Performance.trade_date == ref_date,
                Performance.type.in_(build_types),
                Performance.universe.in_(universes),
                Performance.benchmark.in_(benchmarks),
                Performance.source.in_(sources),
                Performance.portfolio.in_(portfolios)
            )
        )

        self.engine.execute(query)
        df.to_sql(Performance.__table__.name, self.engine, if_exists='append', index=False)

    def upsert_positions(self, ref_date, df):
        universes = df.universe.unique().tolist()
        benchmarks = df.benchmark.unique().tolist()
        build_types = df.type.unique().tolist()
        sources = df.source.unique().tolist()
        portfolios = df.portfolio.unique().tolist()

        query = delete(Positions).where(
            and_(
                Positions.trade_date == ref_date,
                Positions.type.in_(build_types),
                Positions.universe.in_(universes),
                Positions.benchmark.in_(benchmarks),
                Positions.source.in_(sources),
                Positions.portfolio.in_(portfolios)
            )
        )

        self.engine.execute(query)
        df.to_sql(Positions.__table__.name,
                  self.engine,
                  if_exists='append',
                  index=False,
                  dtype={'weight': sa.types.JSON})


if __name__ == '__main__':

    universe = Universe('ss', ['hs300'])

    engine = SqlEngine()

    df = engine.fetch_factor_range(universe,
                                   ['closePrice'],
                                   start_date='2012-12-21',
                                   end_date='2017-12-25')

    print(df)
