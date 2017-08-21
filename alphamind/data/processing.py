# -*- coding: utf-8 -*-
"""
Created on 2017-8-21

@author: cheng.li
"""

from typing import Optional
from typing import List
import numpy as np
from alphamind.data.neutralize import neutralize


def factor_processing(raw_factors: np.ndarray,
                      pre_process: Optional[List]=None,
                      risk_factors: Optional[np.ndarray]=None,
                      post_process: Optional[List]=None) -> np.ndarray:

    new_factors = raw_factors

    if pre_process:
        for p in pre_process:
            new_factors = p(new_factors)

    if risk_factors is not None:
        risk_factors = risk_factors[:, risk_factors.sum(axis=0) != 0]
        new_factors = neutralize(risk_factors, new_factors)

    if post_process:
        for p in post_process:
            new_factors = p(new_factors)

    return new_factors
