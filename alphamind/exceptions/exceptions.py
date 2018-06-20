# -*- coding: utf-8 -*-
"""
Created on 2018-6-12

@author: cheng.li
"""


class PortfolioBuilderException(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)