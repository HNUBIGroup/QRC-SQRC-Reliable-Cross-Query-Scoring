"""Numerical checks for the portable correction API."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from qrc import correct, estimate_tau2


class CorrectionTests(unittest.TestCase):
    def test_additive_recovery(self):
        base = np.array([.1,.5,.2,.6])
        queries = ['a','a','b','b']
        raw = base+np.array([3.,3.,-2.,-2.])
        result = correct(base,raw,queries,tau2=1.)
        np.testing.assert_allclose(result['qrc'],base,atol=1e-14)
        np.testing.assert_allclose(result['sqrc'],base,atol=1e-14)

    def test_zero_heterogeneity_no_sqrc_correction(self):
        base = np.zeros(4)
        raw = np.array([1.,3.,4.,8.])
        result = correct(base,raw,['a','a','b','b'],tau2=0.)
        np.testing.assert_array_equal(result['sqrc'],raw)
        np.testing.assert_allclose(result['qrc'],[-1.,1.,-2.,2.])

    def test_validation_moments(self):
        # Query means 1 and 3: sample between-query variance is 2.
        self.assertAlmostEqual(estimate_tau2([0.]*4,[1.,1.,3.,3.],['v1','v1','v2','v2']),2.)
        self.assertEqual(estimate_tau2([0.]*4,[0.,2.,0.,2.],['v1','v1','v2','v2']),0.)

    def test_reject_invalid_input(self):
        for base,raw,queries in [([],[],[]),([0.],[1.,2.],['a']),([0.],[float('nan')],['a']),([0.],[1.],[None])]:
            with self.assertRaises(ValueError):
                correct(base,raw,queries)
        with self.assertRaises(ValueError):
            correct([0.],[1.],['a'],tau2=-1.)
        with self.assertRaises(ValueError):
            estimate_tau2([0.],[1.],['v'])


if __name__ == '__main__':
    unittest.main()
