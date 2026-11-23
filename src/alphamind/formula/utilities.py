# -*- coding: utf-8 -*-
"""
Created on 2017-11-27

@author: cheng.li
"""

from alphamind.utilities import decode, encode


def encode_formula(formula):
    str_repr = encode(formula)
    return {
        "desc": str_repr,
        "formula_type": formula.__class__.__module__ + "." + formula.__class__.__name__,
        "dependency": formula.fields,
        "window": formula.window,
    }


def decode_formula(str_repr):
    if isinstance(str_repr, dict):
        str_repr = str_repr["desc"]
    return decode(str_repr)
