# -*- coding: utf-8 -*-
# distutils: language = c++
"""
Created on 2017-4-26

@author: cheng.li
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from libc.math cimport fabs
from cpython.mem cimport PyMem_Malloc
from cpython.mem cimport PyMem_Free
from numpy import array
from libcpp.vector cimport vector as cpp_vector
from libcpp.unordered_map cimport unordered_map as cpp_map
from cython.operator cimport dereference as deref

np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

ctypedef long long int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef list groupby(long[:] groups):

    cdef long long length = groups.shape[0]
    cdef cpp_map[long, cpp_vector[int64_t]] group_ids
    cdef long long i
    cdef long curr_tag
    cdef cpp_map[long, cpp_vector[int64_t]].iterator it
    cdef list res = []
    cdef np.ndarray[long long, ndim=1] npy_array
    cdef cpp_vector[int64_t] v
    cdef long long* arr_ptr

    for i in range(length):
        curr_tag = groups[i]
        it = group_ids.find(curr_tag)

        if it == group_ids.end():
            group_ids[curr_tag] = [i]
        else:
            deref(it).second.push_back(i)

    for v in group_ids.values():
        res.append(array(v, dtype=np.int64))

    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef long* group_mapping(long* groups, size_t length, size_t* max_g):
    cdef long *res_ptr = <long*>PyMem_Malloc(length*sizeof(long))
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
            res_ptr[i] = running_tag
            current_hold[curr_tag] = running_tag
        else:
            res_ptr[i] = deref(it).second

    max_g[0] = running_tag
    return res_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double* agg_sum(long* groups, size_t max_g, double* x, size_t length, size_t width):
    cdef double* res_ptr = <double*>PyMem_Malloc((max_g+1)*width*sizeof(double))
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2
    cdef long curr

    for i in range((max_g+1)*width):
        res_ptr[i] = 0.

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
cdef double* agg_abssum(long* groups, size_t max_g, double* x, size_t length, size_t width):
    cdef double* res_ptr = <double*>PyMem_Malloc((max_g+1)*width*sizeof(double))
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2
    cdef long curr

    for i in range((max_g+1)*width):
        res_ptr[i] = 0.

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
cdef double* agg_mean(long* groups, size_t max_g, double* x, size_t length, size_t width):
    cdef double* res_ptr = <double*>PyMem_Malloc((max_g+1)*width*sizeof(double))
    cdef long* bin_count_ptr = <long*>PyMem_Malloc((max_g+1)*sizeof(long))
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2
    cdef long curr

    try:
        for i in range((max_g+1)*width):
            res_ptr[i] = 0.

        for i in range(max_g+1):
            bin_count_ptr[i] = 0

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
    finally:
        PyMem_Free(bin_count_ptr)
    return res_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double* agg_std(long* groups, size_t max_g, double* x, size_t length, size_t width, long ddof=1):
    cdef double* running_sum_square_ptr = <double*>PyMem_Malloc((max_g+1)*width*sizeof(double))
    cdef double* running_sum_ptr = <double*>PyMem_Malloc((max_g+1)*width*sizeof(double))
    cdef long* bin_count_ptr = <long*>PyMem_Malloc((max_g+1)*sizeof(long))
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2
    cdef long curr
    cdef double raw_value

    try:
        for i in range((max_g+1)*width):
            running_sum_square_ptr[i] = 0.
            running_sum_ptr[i] = 0.

        for i in range(max_g+1):
            bin_count_ptr[i] = 0


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
    finally:
        PyMem_Free(running_sum_ptr)
        PyMem_Free(bin_count_ptr)
    return running_sum_square_ptr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[double, ndim=2] transform(long[:] groups, double[:, :] x, str func):
    cdef size_t length = x.shape[0]
    cdef size_t width = x.shape[1]
    cdef size_t* max_g = <size_t*>PyMem_Malloc(sizeof(size_t))
    cdef long* mapped_groups = group_mapping(&groups[0], length, max_g)
    cdef double* res_data_ptr = <double*>PyMem_Malloc(length*width*sizeof(double))
    cdef double* value_data_ptr
    cdef np.ndarray[double, ndim=2] res
    cdef size_t i
    cdef size_t j
    cdef size_t loop_idx1
    cdef size_t loop_idx2

    try:
        if func == 'mean':
            value_data_ptr = agg_mean(mapped_groups, max_g[0], &x[0, 0], length, width)
        elif func == 'std':
            value_data_ptr = agg_std(mapped_groups, max_g[0], &x[0, 0], length, width, ddof=1)
        elif func == 'sum':
            value_data_ptr = agg_sum(mapped_groups, max_g[0], &x[0, 0], length, width)
        elif func =='abssum':
            value_data_ptr = agg_abssum(mapped_groups, max_g[0], &x[0, 0], length, width)
        else:
            raise ValueError('({0}) is not recognized as valid functor'.format(func))

        with nogil:
            for i in range(length):
                loop_idx1 = i*width
                loop_idx2 = mapped_groups[i] * width
                for j in range(width):
                    res_data_ptr[loop_idx1 + j] = value_data_ptr[loop_idx2 + j]
    finally:
        PyMem_Free(value_data_ptr)
        PyMem_Free(mapped_groups)
        PyMem_Free(max_g)
    res = np.PyArray_SimpleNewFromData(2, [length, width], np.NPY_FLOAT64, res_data_ptr)
    PyArray_ENABLEFLAGS(res, np.NPY_OWNDATA)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray[double, ndim=2] aggregate(long[:] groups, double[:, :] x, str func):
    cdef size_t length = x.shape[0]
    cdef size_t width = x.shape[1]
    cdef size_t* max_g = <size_t*>PyMem_Malloc(sizeof(size_t))
    cdef long* mapped_groups = group_mapping(&groups[0], length, max_g)
    cdef np.ndarray[double, ndim=2] res
    cdef double* value_data_ptr

    try:
        if func == 'mean':
            value_data_ptr = agg_mean(mapped_groups, max_g[0], &x[0, 0], length, width)
        elif func == 'std':
            value_data_ptr = agg_std(mapped_groups, max_g[0], &x[0, 0], length, width, ddof=1)
        elif func == 'sum':
            value_data_ptr = agg_sum(mapped_groups, max_g[0], &x[0, 0], length, width)
        elif func =='abssum':
            value_data_ptr = agg_abssum(mapped_groups, max_g[0], &x[0, 0], length, width)
        else:
            raise ValueError('({0}) is not recognized as valid functor'.format(func))

        res = np.PyArray_SimpleNewFromData(2, [max_g[0]+1, width], np.NPY_FLOAT64, value_data_ptr)
        PyArray_ENABLEFLAGS(res, np.NPY_OWNDATA)
    finally:
        PyMem_Free(mapped_groups)
        PyMem_Free(max_g)

    return res