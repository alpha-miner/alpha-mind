# -*- coding: utf-8 -*-
# distutils: language = c++
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
from numpy import zeros
cimport numpy as np
from numpy import array
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef groupby(long[:] groups):

    cdef size_t length = groups.shape[0]
    cdef dict group_ids = {}
    cdef size_t i
    cdef long curr_tag

    for i in range(length):
        curr_tag = groups[i]

        try:
            group_ids[curr_tag].append(i)
        except KeyError:
            group_ids[curr_tag] = [i]

    return [array(v) for v in group_ids.values()]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[long, ndim=1] group_mapping(long[:] groups):
    cdef size_t length = groups.shape[0]
    cdef np.ndarray[long, ndim=1] res= zeros(length, dtype=long)
    cdef dict current_hold = {}
    cdef long curr_tag
    cdef long running_tag = -1
    cdef size_t i

    for i in range(length):
        curr_tag = groups[i]
        try:
            res[i] = current_hold[curr_tag]
        except KeyError:
            running_tag += 1
            res[i] = running_tag
            current_hold[curr_tag] = running_tag

    return res
