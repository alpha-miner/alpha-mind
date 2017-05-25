# -*- coding: utf-8 -*-
"""
Created on 2017-5-25

@author: cheng.li
"""

import numpy as np
from typing import Optional
from typing import List
from alphamind.data.neutralize import neutralize


def factor_processing(raw_factor: np.ndarray,
                      pre_process: Optional[List]=None,
                      risk_factors: Optional[np.ndarray]=None):

    new_factor = raw_factor

    if pre_process:
        for p in pre_process:
            new_factor = p(new_factor)

    if risk_factors is not None:
        new_factor = neutralize(risk_factors, new_factor)

    return new_factor


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