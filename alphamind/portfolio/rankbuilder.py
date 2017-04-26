# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np


def rank_build(er: np.ndarray, use_rank: int, groups: np.ndarray=None):

    if groups is not None:
        pass
    else:
        ordering = np.argsort(er)


if __name__ == '__main__':
    x = np.random.randn(3000)
