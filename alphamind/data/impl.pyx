# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_groups(long[:] groups, long length) nogil:
    cdef long curr_max = 0
    cdef long i
    cdef long curr

    for i in range(length):
        curr = groups[i]
        if curr > curr_max:
            curr_max = curr
    return curr_max

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=2] agg_mean(long[:] groups, double[:, :] x):
    cdef long length = groups.shape[0]
    cdef long width = x.shape[1]
    cdef long max_g = max_groups(groups, length)
    cdef double[:, :] res = np.zeros((max_g+1, width))
    cdef long[:] bin_count = np.zeros(max_g+1, dtype=int)
    cdef long i
    cdef long j
    cdef long curr

    for i in range(length):
        for j in range(width):
            res[groups[i], j] += x[i, j]
        bin_count[groups[i]] += 1

    for i in range(res.shape[0]):
        curr = bin_count[i]
        if curr != 0:
            for j in range(width):
                res[i, j] /= curr
    return np.asarray(res)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim=2] agg_std(long[:] groups, double[:, :] x, long ddof=1):
    cdef long length = groups.shape[0]
    cdef long width = x.shape[1]
    cdef long max_g = max_groups(groups, length)
    cdef double[:, :] running_sum_square = np.zeros((max_g+1, width))
    cdef double[:, :] running_sum = np.zeros((max_g+1, width))
    cdef long[:] bin_count = np.zeros(max_g+1, dtype=int)
    cdef long i
    cdef long j
    cdef long curr
    cdef double raw_value

    for i in range(length):
        for j in range(width):
            raw_value = x[i, j]
            running_sum[groups[i], j] += raw_value
            running_sum_square[groups[i], j] += raw_value * raw_value
        bin_count[groups[i]] += 1

    for i in range(running_sum_square.shape[0]):
        curr = bin_count[i]
        if curr > ddof:
            for j in range(width):
                running_sum_square[i, j] = sqrt((running_sum_square[i, j] - running_sum[i, j] * running_sum[i, j] / curr) / (curr - ddof))
    return np.asarray(running_sum_square)