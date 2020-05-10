# -*- coding: utf-8 -*-
"""
Created on 2017-11-27

@author: cheng.li
"""

from alphamind.utilities import decode
from alphamind.utilities import encode


def encode_formula(formula):
    str_repr = encode(formula)
    return {'desc': str_repr,
            'formula_type': formula.__class__.__module__ + "." + formula.__class__.__name__,
            'dependency': formula.fields,
            'window': formula.window}


def decode_formula(str_repr):
    formula = decode(str_repr)
    return formula


if __name__ == '__main__':
    from PyFin.api import *

    eps_q_res = RES(20, LAST('eps_q') ^ LAST('roe_q'))
    print(eps_q_res)

    str_repr = encode_formula(eps_q_res)
    decoded_formula = decode_formula(str_repr)
    print(decoded_formula)
