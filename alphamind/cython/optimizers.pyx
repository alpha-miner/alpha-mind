# -*- coding: utf-8 -*-
# distutils: language = c++
"""
Created on 2017-7-20

@author: cheng.li
"""

cimport numpy as cnp
import numpy as np
from libcpp.vector cimport vector


cdef extern from "lpoptimizer.hpp" namespace "pfopt":
    cdef cppclass LpOptimizer:
        LpOptimizer(int, int, double*, double*, double*, double*) except +
        vector[double] xValue()
        double feval()
        int status()


cdef class LPOptimizer:

    cdef LpOptimizer* cobj

    def __init__(self,
                 cnp.ndarray[double, ndim=2] cons_matrix,
                 double[:] lbound,
                 double[:] ubound,
                 double[:] objective):
        cdef int n = lbound.shape[0]
        cdef int m = cons_matrix.shape[0]
        cdef double[:] cons = cons_matrix.flatten(order='C');

        self.cobj = new LpOptimizer(n,
                                    m,
                                    &cons[0],
                                    &lbound[0],
                                    &ubound[0],
                                    &objective[0])

    def __del__(self):
        del self.cobj

    def status(self):
        return self.cobj.status()

    def feval(self):
        return self.cobj.feval()

    def x_value(self):
        return np.array(self.cobj.xValue())


cdef extern from "mvoptimizer.hpp" namespace "pfopt":
    cdef cppclass MVOptimizer:
        MVOptimizer(int,
                    double*,
                    double*,
                    double*,
                    double*,
                    int,
                    double*,
                    double*,
                    double*,
                    double) except +
        vector[double] xValue()
        double feval()
        int status()


cdef class QPOptimizer:

    cdef MVOptimizer* cobj

    def __init__(self,
                 double[:] expected_return,
                 cnp.ndarray[double, ndim=2] cov_matrix,
                 double[:] lbound,
                 double[:] ubound,
                 cnp.ndarray[double, ndim=2] cons_matrix,
                 double[:] clbound,
                 double[:] cubound,
                 double risk_aversion=1.0):

        cdef int n = lbound.shape[0]
        cdef int m = cons_matrix.shape[0]
        cdef double[:] cov = cov_matrix.flatten(order='C')
        cdef double[:] cons = cons_matrix.flatten(order='C');

        self.cobj = new MVOptimizer(n,
                                    &expected_return[0],
                                    &cov[0],
                                    &lbound[0],
                                    &ubound[0],
                                    m,
                                    &cons[0],
                                    &clbound[0],
                                    &cubound[0],
                                    risk_aversion)

    def __del__(self):
        del self.cobj

    def feval(self):
        return self.cobj.feval()

    def x_value(self):
        return np.array(self.cobj.xValue())

    def status(self):
        return self.cobj.status()