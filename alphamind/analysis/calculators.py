# -*- coding: utf-8 -*-
"""
Created on 2017-5-18

@author: cheng.li
"""

import pandas as pd


def calculate_turn_over(pos_table: pd.DataFrame) -> pd.DataFrame:
    turn_over_table = {}
    total_factors = pos_table.columns.difference(['code'])
    pos_table.reset_index()

    for name in total_factors:
        pivot_position = pos_table.pivot(values=name, columns='code').fillna(0.)
        turn_over_series = pivot_position.diff().abs().sum(axis=1)
        turn_over_table[name] = turn_over_series.values

    turn_over_table = pd.DataFrame(turn_over_table, index=pos_table.trade_date.unique())
    return turn_over_table[total_factors]
