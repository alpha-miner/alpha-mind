# -*- coding: utf-8 -*-
"""
Created on 2017-4-29

@author: cheng.li
"""

import numpy as np
from numpy import array
cimport numpy as cnp
cimport cython
import cytoolz


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline long index(tuple x):
    return x[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef list groupby(long[:] groups):

    cdef int i
    cdef long d
    cdef list table
    cdef tuple t
    cdef list v
    cdef dict group_dict
    cdef list group_ids

    table = [(d, i) for i, d in enumerate(groups)]
    group_dict = cytoolz.groupby(index, table)
    group_ids = [array([t[1] for t in v]) for v in group_dict.values()]
    return group_ids