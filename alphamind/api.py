# -*- coding: utf-8 -*-
"""
Created on 2017-8-16

@author: cheng.li
"""

from alphamind.data.engines.sqlengine import SqlEngine
from alphamind.data.engines.sqlengine import risk_styles
from alphamind.data.engines.sqlengine import industry_styles
from alphamind.data.engines.sqlengine import macro_styles
from alphamind.analysis.factoranalysis import er_portfolio_analysis
from alphamind.analysis.factoranalysis import factor_analysis
from alphamind.analysis.quantileanalysis import er_quantile_analysis
from alphamind.analysis.quantileanalysis import quantile_analysis
from alphamind.data.engines.universe import Universe
from alphamind.data.engines.utilities import industry_list
from alphamind.data.neutralize import neutralize
from alphamind.data.processing import factor_processing
from alphamind.data.rank import percentile
from alphamind.data.rank import rank
from alphamind.data.standardize import Standardizer
from alphamind.data.standardize import projection
from alphamind.data.standardize import standardize
from alphamind.data.winsorize import NormalWinsorizer
from alphamind.data.winsorize import winsorize_normal
from alphamind.execution.naiveexecutor import NaiveExecutor
from alphamind.execution.pipeline import ExecutionPipeline
from alphamind.execution.targetvolexecutor import TargetVolExecutor
from alphamind.execution.thresholdexecutor import ThresholdExecutor
from alphamind.model import ConstLinearModel
from alphamind.model import LassoRegression
from alphamind.model import LinearRegression
from alphamind.model import LogisticRegression
from alphamind.model import NvSVRModel
from alphamind.model import RandomForestClassifier
from alphamind.model import RandomForestRegressor
from alphamind.model import XGBClassifier
from alphamind.model import XGBRegressor
from alphamind.model import XGBTrainer
from alphamind.model import load_model
from alphamind.model.composer import Composer
from alphamind.model.composer import DataMeta
from alphamind.model.composer import predict_by_model
from alphamind.model.composer import train_model
from alphamind.model.data_preparing import fetch_data_package
from alphamind.model.data_preparing import fetch_predict_phase
from alphamind.model.data_preparing import fetch_train_phase
from alphamind.portfolio.constraints import BoundaryDirection
from alphamind.portfolio.constraints import BoundaryType
from alphamind.portfolio.constraints import Constraints
from alphamind.portfolio.constraints import LinearConstraints
from alphamind.portfolio.constraints import create_box_bounds
from alphamind.portfolio.evolver import evolve_positions
from alphamind.utilities import alpha_logger
from alphamind.utilities import map_freq

__all__ = [
    'SqlEngine',
    'factor_analysis',
    'er_portfolio_analysis',
    'quantile_analysis',
    'er_quantile_analysis',
    'Universe',
    'factor_processing',
    'Constraints',
    'LinearConstraints',
    'BoundaryType',
    'BoundaryDirection',
    'create_box_bounds',
    'evolve_positions',
    'risk_styles',
    'industry_styles',
    'macro_styles',
    'winsorize_normal',
    'NormalWinsorizer',
    'standardize',
    'Standardizer',
    'projection',
    'neutralize',
    'rank',
    'percentile',
    'industry_list',
    'fetch_data_package',
    'fetch_train_phase',
    'fetch_predict_phase',
    'Composer',
    'DataMeta',
    'train_model',
    'predict_by_model',
    'LinearRegression',
    'LassoRegression',
    'ConstLinearModel',
    'LogisticRegression',
    'RandomForestRegressor',
    'RandomForestClassifier',
    'XGBRegressor',
    'XGBClassifier',
    'XGBTrainer',
    'NvSVRModel',
    'load_model',
    'NaiveExecutor',
    'ThresholdExecutor',
    'TargetVolExecutor',
    'ExecutionPipeline',
    'alpha_logger',
    'map_freq'
]
