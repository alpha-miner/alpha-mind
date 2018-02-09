# -*- coding: utf-8 -*-
"""
Created on 2018-2-9

@author: cheng.li
"""

import unittest
from PyFin.api import LAST
from alphamind.data.engines.universe import Universe


class TestUniverse(unittest.TestCase):

    def test_universe_equal(self):
        universe1 = Universe('custom', ['zz500'])
        universe2 = Universe('custom', ['zz500'])
        self.assertEqual(universe1, universe2)

        universe1 = Universe('custom', ['zz500'])
        universe2 = Universe('custom', ['zz800'])
        self.assertNotEqual(universe1, universe2)

        filter_cond = LAST('x') > 1.
        universe1 = Universe('custom', ['zz500'], filter_cond=filter_cond)
        universe2 = Universe('custom', ['zz500'], filter_cond=filter_cond)
        self.assertEqual(universe1, universe2)

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
