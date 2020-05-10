# -*- coding: utf-8 -*-
"""
Created on 2018-5-29

@author: cheng.li
"""

import abc
from typing import List

import pandas as pd


class RiskModel(metaclass=abc.ABCMeta):

    def get_risk_profile(self):
        pass


class FullRiskModel(RiskModel):

    def __init__(self, sec_cov: pd.DataFrame):
        self.codes = sec_cov.index.tolist()
        self.sec_cov = sec_cov.loc[self.codes, self.codes]

    def get_cov(self, codes: List[int] = None):
        if codes:
            return self.sec_cov.loc[codes, codes].values
        else:
            return self.sec_cov.values

    def get_risk_profile(self, codes: List[int] = None):
        return dict(
            cov=self.get_cov(codes),
            factor_cov=None,
            factor_loading=None,
            idsync=None
        )


class FactorRiskModel(RiskModel):

    def __init__(self,
                 factor_cov: pd.DataFrame,
                 risk_exp: pd.DataFrame,
                 idsync: pd.Series):
        self.factor_cov = factor_cov
        self.idsync = idsync
        self.codes = self.idsync.index.tolist()
        self.factor_names = sorted(self.factor_cov.index)
        self.risk_exp = risk_exp.loc[self.codes, self.factor_names]
        self.factor_cov = self.factor_cov.loc[self.factor_names, self.factor_names]
        self.idsync = self.idsync[self.codes]

    def get_risk_exp(self, codes: List[int] = None):
        if codes:
            return self.risk_exp.loc[codes, :].values
        else:
            return self.risk_exp.values

    def get_factor_cov(self):
        return self.factor_cov.values

    def get_idsync(self, codes: List[int] = None):
        if codes:
            return self.idsync[codes].values
        else:
            return self.idsync.values

    def get_risk_profile(self, codes: List[int] = None):
        return dict(
            cov=None,
            factor_cov=self.get_factor_cov(),
            factor_loading=self.get_risk_exp(codes),
            idsync=self.get_idsync(codes)
        )
