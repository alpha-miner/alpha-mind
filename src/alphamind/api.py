# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

from alphamind.analysis.factoranalysis import er_portfolio_analysis, factor_analysis
from alphamind.analysis.quantileanalysis import er_quantile_analysis, quantile_analysis
from alphamind.data.arrow import read_parquet, read_table, write_parquet, write_table
from alphamind.data.calendar import advanceDateByCalendar, bizDatesList, makeSchedule
from alphamind.data.engines.sqlengine import SqlEngine, industry_styles, macro_styles, risk_styles
from alphamind.data.engines.sqlengine.repository import ArrowSQLRepository
from alphamind.data.engines.universe import Universe
from alphamind.data.engines.utilities import industry_list
from alphamind.data.neutralize import neutralize
from alphamind.data.processing import factor_processing
from alphamind.data.rank import percentile, rank
from alphamind.data.standardize import Standardizer, projection, standardize
from alphamind.data.winsorize import NormalWinsorizer, winsorize_normal
from alphamind.execution.naiveexecutor import NaiveExecutor
from alphamind.execution.pipeline import ExecutionPipeline
from alphamind.execution.targetvolexecutor import TargetVolExecutor
from alphamind.execution.thresholdexecutor import ThresholdExecutor
from alphamind.formula import LAST, MA, SHIFT, CSQuantiles, CSRank
from alphamind.model import (
    ConstLinearModel,
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    NvSVRModel,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBClassifier,
    XGBRegressor,
    XGBTrainer,
    load_model,
)
from alphamind.model.composer import Composer, DataMeta, predict_by_model, train_model
from alphamind.model.data_preparing import (
    fetch_data_package,
    fetch_predict_phase,
    fetch_train_phase,
)
from alphamind.portfolio.constraints import (
    BoundaryDirection,
    BoundaryType,
    Constraints,
    LinearConstraints,
    create_box_bounds,
)
from alphamind.portfolio.evolver import evolve_positions
from alphamind.utilities import alpha_logger, map_freq

__all__ = [
    "SqlEngine",
    "factor_analysis",
    "er_portfolio_analysis",
    "quantile_analysis",
    "er_quantile_analysis",
    "Universe",
    "factor_processing",
    "Constraints",
    "LinearConstraints",
    "BoundaryType",
    "BoundaryDirection",
    "create_box_bounds",
    "evolve_positions",
    "risk_styles",
    "industry_styles",
    "macro_styles",
    "winsorize_normal",
    "NormalWinsorizer",
    "standardize",
    "Standardizer",
    "projection",
    "neutralize",
    "rank",
    "percentile",
    "industry_list",
    "fetch_data_package",
    "fetch_train_phase",
    "fetch_predict_phase",
    "Composer",
    "DataMeta",
    "train_model",
    "predict_by_model",
    "LinearRegression",
    "LassoRegression",
    "ConstLinearModel",
    "LogisticRegression",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "XGBRegressor",
    "XGBClassifier",
    "XGBTrainer",
    "NvSVRModel",
    "load_model",
    "NaiveExecutor",
    "ThresholdExecutor",
    "TargetVolExecutor",
    "ExecutionPipeline",
    "alpha_logger",
    "map_freq",
]

__all__ += [
    "ArrowSQLRepository",
    "read_table",
    "write_table",
    "read_parquet",
    "write_parquet",
    "advanceDateByCalendar",
    "bizDatesList",
    "makeSchedule",
    "LAST",
    "SHIFT",
    "MA",
    "CSRank",
    "CSQuantiles",
]
