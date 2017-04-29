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
from libc.math cimport fabs
from libc.stdlib cimport calloc
from libc.stdlib cimport free

np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef long max_groups(long* groups, size_t length) nogil:
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
cdef double* agg_sum(long* groups, double* x, size_t length, size_t width) nogil:
    cdef long max_g = max_groups(groups, length)
    cdef double* res_ptr = <double*>calloc((max_g+1)*width, sizeof(double))
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2
    cdef long curr

    for i in range(length):
        loop_idx1 = i*width
        loop_idx2 = groups[i]*width
        for j in range(width):
            res_ptr[loop_idx2 + j] += x[loop_idx1 + j]
    return res_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double* agg_abssum(long* groups, double* x, size_t length, size_t width) nogil:
    cdef long max_g = max_groups(groups, length)
    cdef double* res_ptr = <double*>calloc((max_g+1)*width, sizeof(double))
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2
    cdef long curr

    for i in range(length):
        loop_idx1 = i*width
        loop_idx2 = groups[i]*width
        for j in range(width):
            res_ptr[loop_idx2 + j] += fabs(x[loop_idx1 + j])
    return res_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double* agg_mean(long* groups, double* x, size_t length, size_t width) nogil:
    cdef long max_g = max_groups(groups, length)
    cdef double* res_ptr = <double*>calloc((max_g+1)*width, sizeof(double))
    cdef long* bin_count_ptr = <long*>calloc(max_g+1, sizeof(int))
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2
    cdef long curr

    for i in range(length):
        loop_idx1 = i*width
        loop_idx2 = groups[i]*width
        for j in range(width):
            res_ptr[loop_idx2 + j] += x[loop_idx1 + j]
        bin_count_ptr[groups[i]] += 1

    for i in range(max_g+1):
        curr = bin_count_ptr[i]
        if curr != 0:
            loop_idx1 = i*width
            for j in range(width):
                res_ptr[loop_idx1 + j] /= curr

    free(bin_count_ptr)
    return res_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double* agg_std(long* groups, double* x, size_t length, size_t width, long ddof=1) nogil:
    cdef long max_g = max_groups(groups, length)
    cdef double* running_sum_square_ptr = <double*>calloc((max_g+1)*width, sizeof(double))
    cdef double* running_sum_ptr = <double*>calloc((max_g+1)*width, sizeof(double))
    cdef long* bin_count_ptr = <long*>calloc(max_g+1, sizeof(int))
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2
    cdef long curr
    cdef double raw_value

    for i in range(length):
        loop_idx1 = i * width
        loop_idx2 = groups[i] * width

        for j in range(width):
            raw_value = x[loop_idx1 + j]
            running_sum_ptr[loop_idx2 + j] += raw_value
            running_sum_square_ptr[loop_idx2 + j] += raw_value * raw_value
        bin_count_ptr[groups[i]] += 1

    for i in range(max_g+1):
        curr = bin_count_ptr[i]
        loop_idx1 = i * width
        if curr != 0:
            for j in range(width):
                loop_idx2 = loop_idx1 + j
                running_sum_square_ptr[loop_idx2] = sqrt((running_sum_square_ptr[loop_idx2] - running_sum_ptr[loop_idx2] * running_sum_ptr[loop_idx2] / curr) / (curr - ddof))

    free(running_sum_ptr)
    free(bin_count_ptr)
    return running_sum_square_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[double, ndim=2] transform(long[:] groups, double[:, :] x, str func):
    cdef size_t length = x.shape[0]
    cdef size_t width = x.shape[1]
    cdef double* res_data_ptr = <double*>calloc(length*width, sizeof(double))
    cdef double* value_data_ptr
    cdef np.ndarray[double, ndim=2] res
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2


    if func == 'mean':
        value_data_ptr = agg_mean(&groups[0], &x[0, 0], length, width)
    elif func == 'std':
        value_data_ptr = agg_std(&groups[0], &x[0, 0], length, width, ddof=1)
    elif func == 'sum':
        value_data_ptr = agg_sum(&groups[0], &x[0, 0], length, width)
    elif func =='abssum':
        value_data_ptr = agg_abssum(&groups[0], &x[0, 0], length, width)

    with nogil:
        for i in range(length):
            loop_idx1 = i*width
            loop_idx2 = groups[i] * width
            for j in range(width):
                res_data_ptr[loop_idx1 + j] = value_data_ptr[loop_idx2 + j]
        free(value_data_ptr)
    res = np.PyArray_SimpleNewFromData(2, [length, width], np.NPY_FLOAT64, res_data_ptr)
    PyArray_ENABLEFLAGS(res, np.NPY_OWNDATA)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[double, ndim=2] aggregate(long[:] groups, double[:, :] x, str func):
    cdef size_t length = x.shape[0]
    cdef size_t width = x.shape[1]
    cdef long max_g = max_groups(&groups[0], length)
    cdef np.ndarray[double, ndim=2] res
    cdef double* value_data_ptr

    if func == 'mean':
        value_data_ptr = agg_mean(&groups[0], &x[0, 0], length, width)
    elif func == 'std':
        value_data_ptr = agg_std(&groups[0], &x[0, 0], length, width, ddof=1)
    elif func == 'sum':
        value_data_ptr = agg_sum(&groups[0], &x[0, 0], length, width)
    elif func =='abssum':
        value_data_ptr = agg_abssum(&groups[0], &x[0, 0], length, width)

    res = np.PyArray_SimpleNewFromData(2, [max_g+1, width], np.NPY_FLOAT64, value_data_ptr)
    PyArray_ENABLEFLAGS(res, np.NPY_OWNDATA)
    return res