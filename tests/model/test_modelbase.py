"""
Created on 2018-2-8

@author: cheng.li
"""

from alphamind.model.linearmodel import ConstLinearModel


class TestModelBase:
    def test_simple_model_features(self):
        model = ConstLinearModel(features=["c", "b", "a"])

        assert ["a", "b", "c"] == model.features
