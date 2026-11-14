"""Compatibility CPU observer regressions for the current two-gates entry."""
import unittest

import numpy as np

from scripts import check_numerics_paper as paper


class FixedBatchTests(unittest.TestCase):
    def test_common_mask_sample_variance_and_separate_peaks(self):
        reference = np.array([0, 4, np.nan, 4, 5, 100])
        candidate = np.array([1, 2, 0, np.inf, 2, -100])
        oracle = np.array([0, 2.5, 0, 3, np.nan, 0])
        obs = paper.observe(reference, candidate, oracle, np.array([1, 2, 1, 1, 1, 1]),
                            np.array([True, True, True, True, True, False]))
        np.testing.assert_array_equal(obs["delta"], [1., -1.])
        result = paper.evaluate(obs, [2, 3], 5.0)
        self.assertEqual(result["sampling"], {"drawn": 6, "in_domain": 5, "valid": 2,
                                               "excluded_domain": 1, "excluded_nonfinite": 3})
        self.assertAlmostEqual(result["bias"]["sample_std"], np.sqrt(2))
        self.assertEqual(result["bias"]["upper"], 5.)
        self.assertEqual(result["magnitude"]["reference_peak"], 1.5)
        self.assertEqual(result["magnitude"]["candidate_peak"], 1.)
        self.assertEqual(result["decision"], "ACCEPT")
        self.assertEqual(paper.evaluate(obs, [2, 3], 4.99)["decision"], "REJECT")

    def test_peak_threshold_and_zero_reference(self):
        for peaks, ratio, decision in (
            ([2., 20.], 10., "ACCEPT"), ([2., 20.01], 10.005, "REJECT"),
            ([0., 0.], 0., "ACCEPT"), ([0., 1.], "+inf", "REJECT"),
            ([1., 0.], 0., "ACCEPT"),
        ):
            with self.subTest(peaks=peaks):
                obs = dict(delta=np.zeros(2), peaks=np.array(peaks), in_domain=np.array(2, np.int64))
                result = paper.evaluate(obs, [1, 2], .05)
                self.assertEqual(result["magnitude"]["ratio"], ratio)
                self.assertEqual(result["decision"], decision)

    def test_insufficient_samples_and_residual_oracle(self):
        reference, candidate = np.array([0., np.inf]), np.array([0., 2.])
        obs = paper.observe(reference, candidate, np.zeros(2), np.ones(2), np.ones(2, dtype=bool),
                            (np.array([.125, np.inf]), np.array([.25, np.nan])))
        np.testing.assert_array_equal(obs["peaks"], [.125, .25])
        self.assertEqual(paper.evaluate(obs, [1, 2], .05)["decision"], "INCONCLUSIVE")


if __name__ == '__main__':
    unittest.main()
