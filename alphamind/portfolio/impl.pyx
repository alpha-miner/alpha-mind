# -*- coding: utf-8 -*-
"""
Created on 2017-4-29

@author: cheng.li
"""

import numpy as np
cimport numpy as np
from numpy import array
cimport cython
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem
from cpython.ref cimport PyObject
from cpython.list cimport PyList_Append


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
cpdef void set_value_bool(unsigned char[:, :] mat, long long[:, :] index):

    cdef size_t length = index.shape[0]
    cdef size_t width = index.shape[1]
    cdef size_t i
    cdef size_t j
    cdef unsigned char* mat_ptr = &mat[0, 0]
    cdef long long* index_ptr = &index[0, 0]
    cdef size_t k

    for i in range(length):
        k = i * width
        for j in range(width):
            mat_ptr[index_ptr[k + j] * width + j] = True


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef void set_value_double(double[:, :] mat, long long[:, :] index, double val):

    cdef size_t length = index.shape[0]
    cdef size_t width = index.shape[1]
    cdef size_t i
    cdef size_t j
    cdef double* mat_ptr = &mat[0, 0]
    cdef long long* index_ptr = &index[0, 0]
    cdef size_t k

    for i in range(length):
        k = i * width
        for j in range(width):
            mat_ptr[index_ptr[k + j] * width + j] = val
