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
        LpOptimizer(vector[double], vector[double], vector[double], vector[double]) except +
        vector[double] xValue()
        double feval()
        int status()


cdef class LPOptimizer:

    cdef LpOptimizer* cobj

    def __init__(self,
                 cnp.ndarray[double, ndim=2] cons_matrix,
                 cnp.ndarray[double] lbound,
                 cnp.ndarray[double] ubound,
                 cnp.ndarray[double] objective):

        self.cobj = new LpOptimizer(cons_matrix.flatten(order='C'),
                                    lbound,
                                    ubound,
                                    objective)

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
        MVOptimizer(vector[double], vector[double], vector[double], vector[double], double) except +
        vector[double] xValue()
        double feval()
        int status()


cdef class QPOptimizer:

    cdef MVOptimizer* cobj

    def __init__(self,
                 cnp.ndarray[double] expected_return,
                 cnp.ndarray[double, ndim=2] cov_matrix,
                 cnp.ndarray[double] lbound,
                 cnp.ndarray[double] ubound,
                 double risk_aversion):

        self.cobj = new MVOptimizer(expected_return,
                                    cov_matrix.flatten(order='C'),
                                    lbound,
                                    ubound,
                                    risk_aversion)

    def __del__(self):
        del self.cobj

    def feval(self):
        return self.cobj.feval()

    def x_value(self):
        return np.array(self.cobj.xValue())

    def status(self):
        return self.cobj.status()