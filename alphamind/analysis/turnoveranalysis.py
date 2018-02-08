# -*- coding: utf-8 -*-
"""
Created on 2018-1-15

@author: cheng.li
"""

import numpy as np
from alphamind.data.standardize import standardize


def factor_turn_over(factor_values: np.ndarray,
                     trade_dates: np.ndarray,
                     codes: np.ndarray,
                     use_standize: bool=True):
    if use_standize:
        factor_values = standardize(factor_values, trade_dates)


if __name__ == '__main__':
    from alphamind.api import *
    engine = SqlEngine()

    factor = 'ep_q'
    freq = '5b'
    start_date = '2017-06-01'
    end_date = '2017-08-01'
    universe = Universe('custom', ['zz500'])


