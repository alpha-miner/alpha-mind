# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

cimport numpy as np
from numpy import zeros
from numpy import asarray
cimport cython
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int max_groups(long* groups, size_t length) nogil:
    cdef long curr_max = 0
    cdef size_t i
    cdef long curr

    for i in range(length):
        curr = groups[i]
        if curr > curr_max:
            curr_max = curr
    return curr_max

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[:, :] agg_mean(long* groups, double* x, size_t length, size_t width):
    cdef long max_g = max_groups(groups, length)
    cdef double[:, :] res = zeros((max_g+1, width))
    cdef double* res_ptr = &res[0, 0]
    cdef long[:] bin_count = zeros(max_g+1, dtype=int)
    cdef long* bin_count_ptr = &bin_count[0]
    cdef size_t i
    cdef size_t j
    cdef long curr

    with nogil:
        for i in range(length):
            for j in range(width):
                res_ptr[groups[i]*width + j] += x[i*width + j]
            bin_count_ptr[groups[i]] += 1

        for i in range(max_g+1):
            curr = bin_count_ptr[i]
            if curr != 0:
                for j in range(width):
                    res_ptr[i*width + j] /= curr
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double[:, :] agg_std(long* groups, double* x, size_t length, size_t width, long ddof=1):
    cdef long max_g = max_groups(groups, length)
    cdef double[:, :] running_sum_square = zeros((max_g+1, width))
    cdef double* running_sum_square_ptr = &running_sum_square[0, 0]
    cdef double[:, :] running_sum = zeros((max_g+1, width))
    cdef double* running_sum_ptr = &running_sum[0, 0]
    cdef long[:] bin_count = zeros(max_g+1, dtype=int)
    cdef long* bin_count_ptr = &bin_count[0]
    cdef size_t i
    cdef size_t j
    cdef long k
    cdef size_t indice
    cdef long curr
    cdef double raw_value

    with nogil:
        for i in range(length):
            k = groups[i]
            for j in range(width):
                raw_value = x[i*width + j]
                running_sum_ptr[k*width + j] += raw_value
                running_sum_square_ptr[k*width + j] += raw_value * raw_value
            bin_count_ptr[k] += 1

        for i in range(max_g+1):
            curr = bin_count_ptr[i]
            if curr != 0:
                for j in range(width):
                    indice = i * width + j
                    running_sum_square_ptr[indice] = sqrt((running_sum_square_ptr[indice] - running_sum_ptr[indice] * running_sum_ptr[indice] / curr) / (curr - ddof))
    return running_sum_square


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[double, ndim=2] transform(long[:] groups, double[:, :] x, str func):

    cdef size_t length = x.shape[0]
    cdef size_t width = x.shape[1]
    cdef double[:, :] res_data = zeros((length, width))
    cdef double* res_data_ptr = &res_data[0, 0]
    cdef double[:, :] value_data = zeros((length, width))
    cdef double* value_data_ptr
    cdef size_t i
    cdef size_t j
    cdef size_t k

    if func == 'mean':
        value_data = agg_mean(&groups[0], &x[0, 0], length, width)
    elif func == 'std':
        value_data = agg_std(&groups[0], &x[0, 0], length, width, ddof=1)

    value_data_ptr = &value_data[0, 0]

    with nogil:
        for i in range(length):
            k = groups[i]
            for j in range(width):
                res_data_ptr[i*width + j] = value_data_ptr[k*width + j]

    return asarray(res_data)