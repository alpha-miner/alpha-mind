# -*- coding: utf-8 -*-
"""
Created on 2018-2-9

@author: cheng.li
"""

import unittest

from alphamind.data.engines.universe import Universe
from alphamind.data.engines.universe import load_universe


class TestUniverse(unittest.TestCase):

    def test_universe_equal(self):
        universe1 = Universe('zz500')
        universe2 = Universe('zz500')
        self.assertEqual(universe1, universe2)

        universe1 = Universe('zz500')
        universe2 = Universe('zz800')
        self.assertNotEqual(universe1, universe2)

    def test_universe_persistence(self):
        universe = Universe('zz500')
        univ_desc = universe.save()
        loaded_universe = load_universe(univ_desc)
        self.assertEqual(universe, loaded_universe)

    def test_universe_arithmic(self):
        universe = Universe('zz500') + Universe('hs300')
        univ_desc = universe.save()
        loaded_universe = load_universe(univ_desc)
        self.assertEqual(universe, loaded_universe)

        universe = Universe('zz500') - Universe('hs300')
        univ_desc = universe.save()
        loaded_universe = load_universe(univ_desc)
        self.assertEqual(universe, loaded_universe)

        universe = Universe('zz500') & Universe('hs300')
        univ_desc = universe.save()
        loaded_universe = load_universe(univ_desc)
        self.assertEqual(universe, loaded_universe)
