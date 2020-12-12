# -*- coding: utf-8 -*-
"""
Created on 2017-8-23

@author: cheng.li
"""

import copy

import pandas as pd
from PyFin.Analysis.SecurityValueHolders import SecurityValueHolder
from PyFin.api import pyFinAssert
from PyFin.api import transform as transform_impl


def factor_translator(factor_pool):
    if not factor_pool:
        return None, None

    if isinstance(factor_pool, str):
        return {factor_pool: factor_pool}, [factor_pool]
    elif isinstance(factor_pool, SecurityValueHolder):
        return {str(factor_pool): factor_pool}, sorted(factor_pool.fields)
    elif isinstance(factor_pool, dict):
        dependency = set()
        for k, v in factor_pool.items():
            pyFinAssert(isinstance(k, str), ValueError,
                        'factor_name {0} should be string.'.format(k))
            pyFinAssert(isinstance(v, SecurityValueHolder) or isinstance(v, str),
                        ValueError,
                        'expression {0} should be a value hodler or a string.'.format(v))

            if isinstance(v, str):
                dependency = dependency.union([v])
            else:
                dependency = dependency.union(v.fields)
        return factor_pool, sorted(dependency)
    elif isinstance(factor_pool, list):
        factor_dict = {}
        dependency = set()
        k = 1
        for i, f in enumerate(factor_pool):
            if isinstance(f, str):
                factor_dict[f] = f
                dependency = dependency.union([f])
            elif isinstance(f, SecurityValueHolder):
                factor_dict[str(f)] = f
                dependency = dependency.union(f.fields)
                k += 1
        return factor_dict, sorted(dependency)
    else:
        raise ValueError('{0} is not in valid format as factors'.format(factor_pool))


class Transformer(object):

    def __init__(self,
                 expressions):
        expression_dict, expression_dependency = \
            factor_translator(copy.deepcopy(expressions))

        if expression_dict:
            self.names = sorted(expression_dict.keys())
            self.expressions = [expression_dict[n] for n in self.names]
            self.dependency = expression_dependency
        else:
            self.names = []
            self.expressions = []
            self.dependency = []

    def transform(self, group_name, data):
        if len(data) > 0:
            transformed_data = transform_impl(data,
                                              self.expressions,
                                              self.names,
                                              group_name,
                                              bool_dropna=False)
            return transformed_data
        else:
            return pd.DataFrame()


if __name__ == '__main__':
    transformer = Transformer(['c', 'a'])
