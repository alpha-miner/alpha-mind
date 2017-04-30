# -*- coding: utf-8 -*-
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from libc.math cimport fabs
from libc.stdlib cimport calloc
from libc.stdlib cimport free
from numpy import array
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem
from cpython.ref cimport PyObject
from cpython.list cimport PyList_Append

np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef inline object _groupby_core(dict d, object key, object item):
    cdef PyObject *obj = PyDict_GetItem(d, key)
    if obj is NULL:
        val = []
        PyList_Append(val, item)
        PyDict_SetItem(d, key, val)
    else:
        PyList_Append(<object>obj, item)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef list groupby(long[:] groups):

    cdef size_t length = groups.shape[0]
    cdef dict group_ids = {}
    cdef size_t i
    cdef long curr_tag

    for i in range(length):
        _groupby_core(group_ids, groups[i], i)

    return [array(v, dtype=np.int64) for v in group_ids.values()]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef long* group_mapping(long* groups, size_t length, size_t* max_g):
    cdef long *res_ptr = <long*>calloc(length, sizeof(int))
    cdef dict current_hold = {}
    cdef long curr_g
    cdef long running_g = -1
    cdef size_t i = 0

    for i in range(length):
        curr_g = groups[i]
        if curr_g not in current_hold:
            running_g += 1
            res_ptr[i] = running_g
            current_hold[curr_g] = running_g
        else:
            res_ptr[i] = current_hold[curr_g]

    max_g[0] = running_g
    return res_ptr


cpdef group_mapping_test(long[:] groups):
    cdef size_t length = groups.shape[0]
    cdef size_t* max_g = <size_t*>calloc(1, sizeof(size_t))
    cdef size_t g_max
    cdef long* mapped_groups = group_mapping(&groups[0], length, max_g)

    res = np.PyArray_SimpleNewFromData(1, [length], np.NPY_INT32, mapped_groups)
    PyArray_ENABLEFLAGS(res, np.NPY_OWNDATA)
    g_max = max_g[0]
    free(max_g)
    return res, g_max


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double* agg_sum(long* groups, size_t max_g, double* x, size_t length, size_t width) nogil:
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
cdef double* agg_abssum(long* groups, size_t max_g, double* x, size_t length, size_t width) nogil:
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
cdef double* agg_mean(long* groups, size_t max_g, double* x, size_t length, size_t width) nogil:
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
cdef double* agg_std(long* groups, size_t max_g, double* x, size_t length, size_t width, long ddof=1) nogil:
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
    cdef size_t* max_g = <size_t*>calloc(1, sizeof(size_t))
    cdef long* mapped_groups = group_mapping(&groups[0], length, max_g)
    cdef double* res_data_ptr = <double*>calloc(length*width, sizeof(double))
    cdef double* value_data_ptr
    cdef np.ndarray[double, ndim=2] res
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2


    if func == 'mean':
        value_data_ptr = agg_mean(mapped_groups, max_g[0], &x[0, 0], length, width)
    elif func == 'std':
        value_data_ptr = agg_std(mapped_groups, max_g[0], &x[0, 0], length, width, ddof=1)
    elif func == 'sum':
        value_data_ptr = agg_sum(mapped_groups, max_g[0], &x[0, 0], length, width)
    elif func =='abssum':
        value_data_ptr = agg_abssum(mapped_groups, max_g[0], &x[0, 0], length, width)

    with nogil:
        for i in range(length):
            loop_idx1 = i*width
            loop_idx2 = mapped_groups[i] * width
            for j in range(width):
                res_data_ptr[loop_idx1 + j] = value_data_ptr[loop_idx2 + j]
        free(value_data_ptr)
        free(mapped_groups)
        free(max_g)
    res = np.PyArray_SimpleNewFromData(2, [length, width], np.NPY_FLOAT64, res_data_ptr)
    PyArray_ENABLEFLAGS(res, np.NPY_OWNDATA)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[double, ndim=2] aggregate(long[:] groups, double[:, :] x, str func):
    cdef size_t length = x.shape[0]
    cdef size_t width = x.shape[1]
    cdef size_t* max_g = <size_t*>calloc(1, sizeof(size_t))
    cdef long* mapped_groups = group_mapping(&groups[0], length, max_g)
    cdef np.ndarray[double, ndim=2] res
    cdef double* value_data_ptr

    if func == 'mean':
        value_data_ptr = agg_mean(mapped_groups, max_g[0], &x[0, 0], length, width)
    elif func == 'std':
        value_data_ptr = agg_std(mapped_groups, max_g[0], &x[0, 0], length, width, ddof=1)
    elif func == 'sum':
        value_data_ptr = agg_sum(mapped_groups, max_g[0], &x[0, 0], length, width)
    elif func =='abssum':
        value_data_ptr = agg_abssum(mapped_groups, max_g[0], &x[0, 0], length, width)

    res = np.PyArray_SimpleNewFromData(2, [max_g[0]+1, width], np.NPY_FLOAT64, value_data_ptr)
    PyArray_ENABLEFLAGS(res, np.NPY_OWNDATA)
    free(mapped_groups)
    free(max_g)
    return res