# -*- coding: utf-8 -*-
"""
Created on 2017-5-25

@author: cheng.li
"""

import numpy as np
from typing import Optional
from typing import List
from alphamind.data.neutralize import neutralize
from alphamind.portfolio.longshortbulder import long_short_build
from alphamind.portfolio.rankbuilder import rank_build
from alphamind.portfolio.percentbuilder import percent_build
from alphamind.portfolio.linearbuilder import linear_build


def factor_processing(raw_factor: np.ndarray,
                      pre_process: Optional[List]=None,
                      risk_factors: Optional[np.ndarray]=None) -> np.ndarray:

    new_factor = raw_factor

    if pre_process:
        for p in pre_process:
            new_factor = p(new_factor)

    if risk_factors is not None:
        new_factor = neutralize(risk_factors, new_factor)

    return new_factor


def build_portfolio(er: np.ndarray,
                    builder: Optional[str]='long_short',
                    **kwargs) -> np.ndarray:

    builder = builder.lower()

    if builder == 'long_short':
        return long_short_build(er, **kwargs)
    elif builder == 'rank':
        return rank_build(er, **kwargs)
    elif builder == 'percent_build':
        return percent_build(er, **kwargs)
    elif builder == 'linear_prog':
        status, _, weight = linear_build(er, **kwargs)
        if status != 'optimal':
            raise ValueError('linear programming optimizer in status: {0}'.format(status))
        else:
            return weight


if __name__ == '__main__':

    from alphamind.data.standardize import standardize
    from alphamind.data.winsorize import winsorize_normal

    raw_factor = np.random.randn(1000, 1)
    pre_process = [winsorize_normal, standardize]

    risk_factors = np.ones((1000, 1))

    new_factor = factor_processing(raw_factor,
                                   pre_process,
                                   risk_factors)

    print(new_factor.sum())