# -*- coding: utf-8 -*-
"""
Created on 2017-4-29

@author: cheng.li
"""

import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef void set_value_bool(unsigned char[:, :] mat, list index, long long[:, :] used_level):

    cdef size_t length = used_level.shape[0]
    cdef size_t width = used_level.shape[1]
    cdef size_t i
    cdef size_t j
    cdef unsigned char* mat_ptr = &mat[0, 0]
    cdef long long* used_level_ptr = &used_level[0, 0]
    cdef size_t k
    cdef size_t l

    for i in range(length):
        k = i * width
        for j in range(width):
            l =  index[used_level_ptr[k + j]]
            mat_ptr[l * width + j] = True


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
