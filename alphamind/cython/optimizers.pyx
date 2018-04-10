# -*- coding: utf-8 -*-
# distutils: language = c++
"""
Created on 2017-7-20

@author: cheng.li
"""

cimport numpy as cnp
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np


cdef extern from "lpoptimizer.hpp" namespace "pfopt":
    cdef cppclass LpOptimizer:
        LpOptimizer(int, int, double*, double*, double*, double*, string) except +
        vector[double] xValue()
        double feval()
        int status()


cdef class LPOptimizer:

    cdef LpOptimizer* cobj
    cdef int n
    cdef int m

    def __cinit__(self,
                  cnp.ndarray[double, ndim=2] cons_matrix,
                  double[:] lbound,
                  double[:] ubound,
                  double[:] objective,
                  str method='simplex'):
        self.n = lbound.shape[0]
        self.m = cons_matrix.shape[0]
        py_bytes = method.encode('ascii')
        cdef string c_str = py_bytes
        cdef double[:] cons = cons_matrix.flatten(order='C');

        self.cobj = new LpOptimizer(self.n,
                                    self.m,
                                    &cons[0],
                                    &lbound[0],
                                    &ubound[0],
                                    &objective[0],
                                    c_str)

    def __dealloc__(self):
        del self.cobj

    def status(self):
        return self.cobj.status()

    def feval(self):
        return self.cobj.feval()

    def x_value(self):
        return np.array(self.cobj.xValue())


cdef extern from "tvoptimizer.hpp" namespace "pfopt":
    cdef cppclass TVOptimizer:
        TVOptimizer(int,
                    double*,
                    double*,
                    double*,
                    double*,
                    int,
                    double*,
                    double*,
                    double*,
                    double,
                    double) except +
        vector[double] xValue()
        double feval()
        int status()


cdef class CVOptimizer:
    cdef TVOptimizer* cobj
    cdef int n
    cdef int m

    def __cinit__(self,
                  double[:] expected_return,
                  cnp.ndarray[double, ndim=2] cov_matrix,
                  double[:] lbound,
                  double[:] ubound,
                  cnp.ndarray[double, ndim=2] cons_matrix=None,
                  double[:] clbound=None,
                  double[:] cubound=None,
                  double target_low=0.0,
                  double target_high=1.0):

        self.n = lbound.shape[0]
        self.m = 0
        cdef double[:] cov = cov_matrix.flatten(order='C')
        cdef double[:] cons

        if cons_matrix is not None:
            self.m = cons_matrix.shape[0]
            cons = cons_matrix.flatten(order='C');

            self.cobj = new TVOptimizer(self.n,
                                        &expected_return[0],
                                        &cov[0],
                                        &lbound[0],
                                        &ubound[0],
                                        self.m,
                                        &cons[0],
                                        &clbound[0],
                                        &cubound[0],
                                        target_low,
                                        target_high)
        else:
            self.cobj = new TVOptimizer(self.n,
                                        &expected_return[0],
                                        &cov[0],
                                        &lbound[0],
                                        &ubound[0],
                                        0,
                                        NULL,
                                        NULL,
                                        NULL,
                                        target_low,
                                        target_high)

    def __dealloc__(self):
        del self.cobj

    def feval(self):
        return self.cobj.feval()

    def x_value(self):
        return np.array(self.cobj.xValue())

    def status(self):
        return self.cobj.status()


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


cdef extern from "qpalglib.hpp" namespace "pfopt":
    cdef cppclass QPAlglib:
        QPAlglib(int,
                 double*,
                 double*,
                 double*,
                 double*,
                 double) except +
        vector[double] xValue()
        int status()


cdef class QPOptimizer:

    cdef MVOptimizer* cobj
    cdef QPAlglib* cobj2
    cdef cnp.ndarray er
    cdef cnp.ndarray cov
    cdef double risk_aversion
    cdef int n
    cdef int m

    def __cinit__(self,
                 double[:] expected_return,
                 cnp.ndarray[double, ndim=2] cov_matrix,
                 double[:] lbound,
                 double[:] ubound,
                 cnp.ndarray[double, ndim=2] cons_matrix=None,
                 double[:] clbound=None,
                 double[:] cubound=None,
                 double risk_aversion=1.0):

        self.n = lbound.shape[0]
        self.m = 0
        self.er = np.array(expected_return)
        self.cov = np.array(cov_matrix)
        self.risk_aversion = risk_aversion
        cdef double[:] cov = cov_matrix.flatten(order='C')
        cdef double[:] cons

        if cons_matrix is not None:
            self.m = cons_matrix.shape[0]
            cons = cons_matrix.flatten(order='C')

            self.cobj = new MVOptimizer(self.n,
                                        &expected_return[0],
                                        &cov[0],
                                        &lbound[0],
                                        &ubound[0],
                                        self.m,
                                        &cons[0],
                                        &clbound[0],
                                        &cubound[0],
                                        risk_aversion)
        else:
            self.cobj2 = new QPAlglib(self.n,
                                      &expected_return[0],
                                      &cov[0],
                                      &lbound[0],
                                      &ubound[0],
                                      risk_aversion)

    def __dealloc__(self):
        if self.cobj:
            del self.cobj
        else:
            del self.cobj2

    def feval(self):
        if self.cobj:
            return self.cobj.feval()
        else:
            x = np.array(self.cobj2.xValue())
            return 0.5 * self.risk_aversion * x @ self.cov @ x - self.er @ x

    def x_value(self):
        if self.cobj:
            return np.array(self.cobj.xValue())
        else:
            return np.array(self.cobj2.xValue())

    def status(self):
        if self.cobj:
            return self.cobj.status()
        else:
            status = self.cobj2.status()

            if 1 <= status <= 4:
                return 0
            else:
                return status
