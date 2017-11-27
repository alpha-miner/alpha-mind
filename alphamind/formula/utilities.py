# -*- coding: utf-8 -*-
"""
Created on 2017-11-27

@author: cheng.li
"""

import pickle
import base64


def encode_formula(formula):
    encoded = base64.encodebytes(pickle.dumps(formula))
    str_repr = encoded.decode('ascii')
    return {'desc': str_repr,
            'formula_type': formula.__class__.__module__ + "." + formula.__class__.__name__,
            'dependency': formula.fields,
            'window': formula.window}


def decode_formula(str_repr):
    encoded = str_repr.encode('ascii')
    formula = pickle.loads(base64.decodebytes(encoded))
    return formula


if __name__ == '__main__':
    from PyFin.api import *

    eps_q_res = RES(20, LAST('eps_q') ^ LAST('roe_q'))
    print(eps_q_res)

    str_repr = encode_formula(eps_q_res)
    decoded_formula = decode_formula(str_repr)
    print(decoded_formula)