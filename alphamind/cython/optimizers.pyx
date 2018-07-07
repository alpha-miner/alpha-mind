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
from PyFin.api import pyFinAssert


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
                    int,
                    double*,
                    double*,
                    double*,
                    string) except +
        vector[double] xValue()
        double feval()
        int status()


cdef class CVOptimizer:
    cdef TVOptimizer* cobj
    cdef int n
    cdef int m
    cdef int f

    def __cinit__(self,
                  double[:] expected_return,
                  cnp.ndarray[double, ndim=2] cov_matrix,
                  double[:] lbound,
                  double[:] ubound,
                  cnp.ndarray[double, ndim=2] cons_matrix=None,
                  double[:] clbound=None,
                  double[:] cubound=None,
                  double target_vol=1.0,
                  cnp.ndarray[double, ndim=2] factor_cov_matrix=None,
                  cnp.ndarray[double, ndim=2] factor_loading_matrix=None,
                  double[:] idsync_risk=None,
                  str linear_solver="ma27"):

        self.n = lbound.shape[0]
        self.m = 0
        self.f = factor_cov_matrix.shape[0] if factor_cov_matrix is not None else 0
        cdef double[:] cov = cov_matrix.flatten(order='C') if cov_matrix is not None else None
        cdef double[:] cons
        cdef double[:] factor_cov = factor_cov_matrix.flatten(order='C') if factor_cov_matrix is not None else None
        cdef double[:] factor_loading = factor_loading_matrix.flatten(order='C') if factor_loading_matrix is not None else None

        if cons_matrix is not None:
            self.m = cons_matrix.shape[0]
            cons = cons_matrix.flatten(order='C')

            self.cobj = new TVOptimizer(self.n,
                                        &expected_return[0],
                                        &cov[0] if cov is not None else NULL,
                                        &lbound[0],
                                        &ubound[0],
                                        self.m,
                                        &cons[0],
                                        &clbound[0],
                                        &cubound[0],
                                        target_vol,
                                        self.f,
                                        &factor_cov[0] if factor_cov is not None else NULL,
                                        &factor_loading[0] if factor_loading is not None else NULL,
                                        &idsync_risk[0] if idsync_risk is not None else NULL,
                                        bytes(linear_solver, encoding='utf8'))
        else:
            self.cobj = new TVOptimizer(self.n,
                                        &expected_return[0],
                                        &cov[0] if cov is not None else NULL,
                                        &lbound[0],
                                        &ubound[0],
                                        0,
                                        NULL,
                                        NULL,
                                        NULL,
                                        target_vol,
                                        self.f,
                                        &factor_cov[0] if factor_cov is not None else NULL,
                                        &factor_loading[0] if factor_loading is not None else NULL,
                                        &idsync_risk[0] if idsync_risk is not None else NULL,
                                        bytes(linear_solver, encoding='utf8'))

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
                    double,
                    int,
                    double*,
                    double*,
                    double*,
                    string) except +
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
    cdef cnp.ndarray er
    cdef cnp.ndarray cov
    cdef double risk_aversion
    cdef int n
    cdef int m
    cdef int f

    def __cinit__(self,
                  double[:] expected_return,
                  cnp.ndarray[double, ndim=2] cov_matrix,
                  double[:] lbound,
                  double[:] ubound,
                  cnp.ndarray[double, ndim=2] cons_matrix=None,
                  double[:] clbound=None,
                  double[:] cubound=None,
                  double risk_aversion=1.0,
                  cnp.ndarray[double, ndim=2] factor_cov_matrix=None,
                  cnp.ndarray[double, ndim=2] factor_loading_matrix=None,
                  double[:] idsync_risk=None,
                  str linear_solver='ma27'):

        self.n = lbound.shape[0]
        self.m = 0
        self.f = factor_cov_matrix.shape[0] if factor_cov_matrix is not None else 0
        self.er = np.array(expected_return)
        self.cov = np.array(cov_matrix)
        self.risk_aversion = risk_aversion
        cdef double[:] cov = cov_matrix.flatten(order='C') if cov_matrix is not None else None
        cdef double[:] cons
        cdef double[:] factor_cov = factor_cov_matrix.flatten(order='C') if factor_cov_matrix is not None else None
        cdef double[:] factor_loading = factor_loading_matrix.flatten(order='C') if factor_loading_matrix is not None else None

        if cons_matrix is not None:
            self.m = cons_matrix.shape[0]
            cons = cons_matrix.flatten(order='C')

            self.cobj = new MVOptimizer(self.n,
                                        &expected_return[0],
                                        &cov[0] if cov is not None else NULL,
                                        &lbound[0],
                                        &ubound[0],
                                        self.m,
                                        &cons[0],
                                        &clbound[0],
                                        &cubound[0],
                                        risk_aversion,
                                        self.f,
                                        &factor_cov[0] if factor_cov is not None else NULL,
                                        &factor_loading[0] if factor_loading is not None else NULL,
                                        &idsync_risk[0] if idsync_risk is not None else NULL,
                                        bytes(linear_solver, encoding='utf8'))
        else:
            self.cobj = new MVOptimizer(self.n,
                                        &expected_return[0],
                                        &cov[0] if cov is not None else NULL,
                                        &lbound[0],
                                        &ubound[0],
                                        self.m,
                                        NULL,
                                        NULL,
                                        NULL,
                                        risk_aversion,
                                        self.f,
                                        &factor_cov[0] if factor_cov is not None else NULL,
                                        &factor_loading[0] if factor_loading is not None else NULL,
                                        &idsync_risk[0] if idsync_risk is not None else NULL,
                                        bytes(linear_solver, encoding='utf8'))

    def __dealloc__(self):
        del self.cobj

    def feval(self):
        return self.cobj.feval()

    def x_value(self):
        return np.array(self.cobj.xValue())

    def status(self):
        return self.cobj.status()
