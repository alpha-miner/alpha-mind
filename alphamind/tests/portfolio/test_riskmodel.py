# -*- coding: utf-8 -*-
"""
Created on 2018-5-29

@author: cheng.li
"""

import unittest
import numpy as np
import pandas as pd
from alphamind.portfolio.riskmodel import FullRiskModel
from alphamind.portfolio.riskmodel import FactorRiskModel


class TestRiskModel(unittest.TestCase):

    def setUp(self):
        self.factor_cov = pd.DataFrame([[0.5, -0.3], [-0.3, 0.7]], columns=['a', 'b'], index=['a', 'b'])
        self.risk_exp = pd.DataFrame([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]], columns=['a', 'b'], index=[1, 2, 3])
        self.idsync = pd.Series([0.1, 0.3, 0.2], index=[1, 2, 3])
        self.sec_cov = self.risk_exp.values @ self.factor_cov.values @ self.risk_exp.values.T \
                       + np.diag(self.idsync.values)
        self.sec_cov = pd.DataFrame(self.sec_cov, columns=[1, 2, 3], index=[1, 2, 3])

    def test_full_risk_model(self):
        self.assertEqual(self.sec_cov.shape, (3, 3))
        model = FullRiskModel(self.sec_cov)

        codes = [1, 2]
        res = model.get_cov(codes)
        np.testing.assert_array_almost_equal(res, self.sec_cov.loc[codes, codes].values)

        res = model.get_cov()
        np.testing.assert_array_almost_equal(res, self.sec_cov.values)

    def test_factor_risk_model(self):
        self.assertEqual(self.factor_cov.shape, (2, 2))
        self.assertEqual(self.risk_exp.shape, (3, 2))
        self.assertEqual(self.idsync.shape, (3,))

        model = FactorRiskModel(self.factor_cov,
                                self.risk_exp,
                                self.idsync)

        res = model.get_factor_cov()
        np.testing.assert_array_almost_equal(res, self.factor_cov.values)

        codes = [1, 3]
        res = model.get_risk_exp(codes)
        np.testing.assert_array_almost_equal(res, self.risk_exp.loc[codes, :])
        res = model.get_idsync(codes)
        np.testing.assert_array_almost_equal(res, self.idsync[codes])

        res = model.get_risk_exp()
        np.testing.assert_array_almost_equal(res, self.risk_exp)
        res = model.get_idsync()
        np.testing.assert_array_almost_equal(res, self.idsync)


