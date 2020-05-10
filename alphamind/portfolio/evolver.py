# -*- coding: utf-8 -*-
"""
Created on 2017-11-23

@author: cheng.li
"""

import numpy as np


def evolve_positions(positions: np.ndarray, dx_ret: np.ndarray) -> np.ndarray:
    # assume return is log return

    simple_return = np.exp(dx_ret)
    evolved_positions = positions * simple_return
    leverage = np.abs(positions).sum()
    evolved_positions = evolved_positions * leverage / np.abs(evolved_positions).sum()
    return evolved_positions
