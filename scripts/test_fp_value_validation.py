"""Validation must retain bit mismatches and reject stale model references."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments/floating_point/value_validation/run_scalar.py'
spec = importlib.util.spec_from_file_location('fp_value_validation_runner', PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ValueValidationTests(unittest.TestCase):
    def reference(self):
        return {'schema': 1, 'source_sha256': runner.model_hashes(), 'cases': [
            {'op': 'add', 'a': 0x3f800000, 'b': 0x33000000, 'expected': 0x3f800000}]}

    def test_exact_source_binding_required(self):
        data = self.reference()
        self.assertEqual(len(runner.validate_reference(data)), 1)
        data['source_sha256'][next(iter(data['source_sha256']))] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source binding'):
            runner.validate_reference(data)

    def test_nan_payload_mismatch_remains_failure(self):
        cases = [{'op': 'add', 'a': 0x7f800001, 'b': 0, 'expected': 0x7fc00000}]
        differences = runner.compare(cases, [0x7fc00001])
        self.assertEqual(len(differences), 1)
        self.assertTrue(differences[0]['both_results_nan'])
        self.assertEqual(differences[0]['actual'], 0x7fc00001)

    def test_signed_zero_is_not_equal_by_numeric_projection(self):
        cases = [{'op': 'neg', 'a': 0, 'b': 0, 'expected': 0x80000000}]
        self.assertEqual(runner.compare(cases, [0x80000000]), [])
        self.assertEqual(len(runner.compare(cases, [0])), 1)

    def test_empty_and_truncated_results_rejected(self):
        data = self.reference()
        with self.assertRaisesRegex(ValueError, 'output length'):
            runner.compare(data['cases'], [])
        data['cases'] = []
        with self.assertRaisesRegex(ValueError, 'no cases'):
            runner.validate_reference(data)

    def test_nonwords_and_unknown_operations_rejected(self):
        for key, value in [('expected', 2**32), ('a', -1), ('b', True), ('op', 'identity')]:
            data = copy.deepcopy(self.reference())
            data['cases'][0][key] = value
            with self.assertRaises(ValueError):
                runner.validate_reference(data)


if __name__ == '__main__':
    unittest.main()
