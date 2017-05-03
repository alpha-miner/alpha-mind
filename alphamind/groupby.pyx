# -*- coding: utf-8 -*-
# distutils: language = c++
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
from numpy import zeros
cimport numpy as np
cimport cython
from libcpp.vector cimport vector as cpp_vector
from libcpp.unordered_map cimport unordered_map as cpp_map
from cython.operator cimport dereference as deref


ctypedef long long int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef groupby(long[:] groups):

    cdef long long length = groups.shape[0]
    cdef cpp_map[long, cpp_vector[int64_t]] group_ids
    cdef long long i
    cdef long curr_tag
    cdef cpp_map[long, cpp_vector[int64_t]].iterator it
    cdef np.ndarray[long long, ndim=1] npy_array

    for i in range(length):
        curr_tag = groups[i]
        it = group_ids.find(curr_tag)

        if it == group_ids.end():
            group_ids[curr_tag] = [i]
        else:
            deref(it).second.push_back(i)

    return [np.array(v) for v in group_ids.values()]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[int, ndim=1] group_mapping(long[:] groups):
    cdef size_t length = groups.shape[0]
    cdef np.ndarray[int, ndim=1] res= zeros(length, dtype=int)
    cdef cpp_map[long, long] current_hold
    cdef long curr_tag
    cdef long running_tag = -1
    cdef size_t i = 0
    cdef cpp_map[long, long].iterator it

    for i in range(length):
        curr_tag = groups[i]
        it = current_hold.find(curr_tag)
        if it == current_hold.end():
            running_tag += 1
            res[i] = running_tag
            current_hold[curr_tag] = running_tag
        else:
            res[i] = deref(it).second

    return res
