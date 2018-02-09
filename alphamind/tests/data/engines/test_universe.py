# -*- coding: utf-8 -*-
"""
Created on 2018-2-9

@author: cheng.li
"""

import unittest
from PyFin.api import LAST
from alphamind.data.engines.universe import Universe


class TestUniverse(unittest.TestCase):

    def test_universe_persistence(self):
        universe = Universe('custom', ['zz500'])
        univ_desc = universe.save()
        loaded_universe = Universe.load(univ_desc)

        self.assertEqual(universe.name, loaded_universe.name)
        self.assertListEqual(universe.base_universe, loaded_universe.base_universe)

        universe = Universe('custom', ['zz500'], filter_cond=LAST('x') > 1.)
        univ_desc = universe.save()
        loaded_universe = Universe.load(univ_desc)

        self.assertEqual(universe.name, loaded_universe.name)
        self.assertListEqual(universe.base_universe, loaded_universe.base_universe)
        self.assertEqual(universe.filter_cond, loaded_universe.filter_cond)