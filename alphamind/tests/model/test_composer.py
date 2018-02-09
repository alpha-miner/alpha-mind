# -*- coding: utf-8 -*-
"""
Created on 2018-2-9

@author: cheng.li
"""

import unittest
from alphamind.data.engines.universe import Universe
from alphamind.model.composer import DataMeta
from alphamind.model.composer import Composer


class TestComposer(unittest.TestCase):

    def test_data_meta_persistence(self):

        freq = '5b'
        universe = Universe('custom', ['zz800'])
        batch = 4
        neutralized_risk = ['SIZE']
        risk_model = 'long'
        pre_process = ['standardize', 'winsorize_normal']
        post_process = ['standardize', 'winsorize_normal']
        warm_start = 2
        data_source = 'postgresql://user:pwd@server/dummy'

        data_meta = DataMeta(freq=freq,
                             universe=universe,
                             batch=batch,
                             neutralized_risk=neutralized_risk,
                             risk_model=risk_model,
                             pre_process=pre_process,
                             post_process=post_process,
                             warm_start=warm_start,
                             data_source=data_source)

        data_desc = data_meta.save()

        loaded_data = DataMeta.load(data_desc)
        self.assertEqual(data_meta.freq, loaded_data.freq)
        self.assertEqual(data_meta.universe, loaded_data.universe)
        self.assertEqual(data_meta.batch, loaded_data.batch)
        self.assertEqual(data_meta.neutralized_risk, loaded_data.neutralized_risk)
        self.assertEqual(data_meta.risk_model, loaded_data.risk_model)
        self.assertEqual(data_meta.pre_process, loaded_data.pre_process)
        self.assertEqual(data_meta.post_process, loaded_data.post_process)
        self.assertEqual(data_meta.warm_start, loaded_data.warm_start)
        self.assertEqual(data_meta.data_source, loaded_data.data_source)

    def test_composer_persistence(self):
        pass



